import paddle
from paddle.vision.datasets import VOC2012
from paddle.vision.transforms import Normalize

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()

    def forward(self, image, label):
        return paddle.sum(image), label

normalize = Normalize(mean=[0.5, 0.5, 0.5],
                      std=[0.5, 0.5, 0.5],
                      data_format='HWC')
voc2012 = VOC2012(mode='train', transform=normalize, backend='cv2')

for i in range(10):
    image, label= voc2012[i]
    image = paddle.cast(paddle.to_tensor(image), 'float32')
    label = paddle.to_tensor(label)

    model = SimpleNet()
    image, label= model(image, label)
    print(image.numpy().shape, label.numpy().shape)