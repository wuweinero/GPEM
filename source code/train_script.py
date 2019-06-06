import numpy as np
from panicle_net_class import PanicleNet

NET_INPUT_SIZE=80
batch_size=16
epoch=10

net=PanicleNet(NET_INPUT_SIZE,TRAIN_FLAG=True)
# X,Y由data_augmentation_for_FCN.py生成
X = np.load('F:/代码临时/train_images.npy')
Y = np.load('F:/代码临时/train_groundtruth.npy')

net.train(X,Y,batch_size,epoch)

print("training is completed")
