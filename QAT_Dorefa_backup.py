import paddle
import paddleslim as slim
import numpy as np
import paddle.fluid as fluid
from paddle.optimizer import Adam
from paddle.fluid.data import data
from paddle.fluid.layer_helper import LayerHelper
paddle.enable_static()


# 采用cpu训练则取False
USE_GPU = True
model = slim.models.MobileNet()
train_program = paddle.static.Program()
startup = paddle.static.Program()
with paddle.static.program_guard(train_program, startup):
    image = paddle.static.data(
        name='image', shape=[None, 1, 28, 28], dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    gt = paddle.reshape(label, [-1, 1])
    out = model.net(input=image, class_dim=10)
    cost = paddle.nn.functional.loss.cross_entropy(input=out, label=gt)
    avg_cost = paddle.mean(x=cost)
    acc_top1 = paddle.metric.accuracy(input=out, label=gt, k=1)
    acc_top5 = paddle.metric.accuracy(input=out, label=gt, k=5)
    opt = paddle.optimizer.Momentum(0.01, 0.9)
    opt.minimize(avg_cost)

place = paddle.CUDAPlace(0) if USE_GPU else paddle.CPUPlace()
exe = paddle.static.Executor(place)
exe.run(startup)
scope = paddle.static.global_scope()
val_program = train_program.clone(for_test=True)


import paddle.vision.transforms as T
transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
# Paddle框架的paddle.vision.dataset包定义了MNIST数据的下载和读取
train_dataset = paddle.vision.datasets.MNIST(
    mode="train", backend="cv2", transform=transform)
test_dataset = paddle.vision.datasets.MNIST(
    mode="test", backend="cv2", transform=transform)
train_loader = paddle.io.DataLoader(
    train_dataset,
    places=place,
    feed_list=[image, label],
    drop_last=True,
    batch_size=64,
    return_list=False,
    shuffle=True)
test_loader = paddle.io.DataLoader(
    test_dataset,
    places=place,
    feed_list=[image, label],
    drop_last=True,
    batch_size=64,
    return_list=False,
    shuffle=False)

# 训练和测试
outputs = [acc_top1.name, acc_top5.name, avg_cost.name]
def train(prog):
    iter = 0
    for data in train_loader():
        acc1, acc5, loss = exe.run(prog, feed=data, fetch_list=outputs)
        if iter % 100 == 0:
            print('train iter={}, top1={}, top5={}, loss={}'.format(iter, acc1.mean(), acc5.mean(), loss.mean()))
        iter += 1

def test(prog):
    iter = 0
    res = [[], []]
    for data in test_loader():
        acc1, acc5, loss = exe.run(prog, feed=data, fetch_list=outputs)
        if iter % 100 == 0:
            print('test iter={}, top1={}, top5={}, loss={}'.format(iter, acc1.mean(), acc5.mean(), loss.mean()))
        res[0].append(acc1.mean())
        res[1].append(acc5.mean())
        iter += 1
    print('final test result top1={}, top5={}'.format(np.array(res[0]).mean(), np.array(res[1]).mean()))


# train(train_program)
# test(val_program)



quant_config = {
    'weight_bits': 8,
    'activation_bits': 8,
    'not_quant_pattern': ['skip_quant'],
    'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul'],
    'dtype': 'int8',
    'window_size': 10000,
    'moving_rate': 0.9
}
quant_exe = paddle.static.Executor(place)
fluid_exe = fluid.Executor(fluid.CPUPlace())

def _load_variable_data(scope, var_name):
    '''
    Load variable value from scope
    '''
    var_node = scope.find_var(var_name)
    assert var_node is not None, \
        "Cannot find " + var_name + " in scope."
    return np.array(var_node.get_tensor())

def _weight_dorefa_quantize_func(in_node):
    '''
    Use Dorefa method to quantize weight.
    '''
    weight_bits = quant_config['weight_bits']
    var_name = in_node.name[0: len(in_node.name) - 10]
    out_node_name = var_name + '_tmp_output'

    # derofa量化
    input = _load_variable_data(scope, var_name)
    output_quant_mid = np.tanh(input)
    output_quant_mid = output_quant_mid / 2 / np.max(np.abs(output_quant_mid)) + 0.5
    scale = 1 / float((1 << (weight_bits - 1)) - 1)
    output_quant_mid = np.round(output_quant_mid / scale) * scale   # STE
    output_quant = 2 * output_quant_mid - 1

    out_node = data(
        name=out_node_name,
        shape=in_node.shape,
        dtype='float32'
    )
    fluid_exe.run(fluid.default_main_program(),
                  feed={
                    in_node.name: input,
                    out_node.name: output_quant
                  })
    '''
    helper = LayerHelper("fake_quantize_dorefa", **locals())
    helper.append_op(
        type="fake_quantize_dorefa",
        inputs={"X": in_node},
        attrs={},
        outputs={"Out": out_node})
    '''
    '''
    tmp_program = Program()
    startup_program = Program()
    with program_guard(tmp_program, startup_program):
        in_node_tmp = data(
            name=in_node.name,
            shape=in_node.shape,
            dtype='float32'
        )
        out_node_tmp = data(
            name=out_node_name,
            shape=in_node.shape,
            dtype='float32'
        )
    fluid_exe.run(startup_program)
    out = fluid_exe.run(tmp_program,
                  feed={
                    in_node_tmp.name: input,
                    out_node_tmp.name: output
                  },
                  fetch_list=[in_node_tmp.name])
    print(out)
    in_node = in_node_tmp
    out_node = out_node_tmp
    '''
    return out_node


quant_program = slim.quant.quant_aware(train_program,
                                       exe.place,
                                       quant_config,
                                       scope,
                                       for_test=False,
                                       weight_quantize_func=_weight_dorefa_quantize_func,
                                       optimizer_func=Adam,
                                       executor=quant_exe)
val_quant_program = slim.quant.quant_aware(val_program,
                                           exe.place,
                                           quant_config,
                                           scope,
                                           for_test=True,
                                           weight_quantize_func=_weight_dorefa_quantize_func,
                                           optimizer_func=Adam,
                                           executor=quant_exe)
# 量化后测试，并与前测试比较精度
train(quant_program)
test(val_quant_program)