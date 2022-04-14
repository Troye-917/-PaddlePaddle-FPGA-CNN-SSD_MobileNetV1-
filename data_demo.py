import paddle
import paddle.fluid as fluid
import numpy as np
paddle.enable_static()
# Creates a variable with fixed size [3, 2, 1]
# User can only feed data of the same shape to x
x = fluid.data(name='x', shape=[3, 2, 1], dtype='float32')

# Creates a variable with changable batch size -1.
# Users can feed data of any batch size into y,
# but size of each data sample has to be [2, 1]
y = fluid.data(name='y', shape=[-1, 2, 1], dtype='float32')

z = x + y

# In this example, we will feed x and y with np-ndarry "1"
# and fetch z, like implementing "1 + 1 = 2" in PaddlePaddle
feed_data = np.zeros(shape=[3, 2, 1], dtype=np.float32)

exe = fluid.Executor(fluid.CPUPlace())
out = exe.run(fluid.default_main_program(),
              feed={
                  'x': feed_data,
                  'y': feed_data
              },
              fetch_list=[z.name])

# np-ndarray of shape=[3, 2, 1], dtype=float32, whose elements are 2
#print(out)
print(out[0][1])