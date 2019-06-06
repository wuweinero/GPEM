# panicle net
## 训练方法
### 1.使用dataset_generator.py生成训练集，需要定义的参数为：
    IMG_PATH = 'F:/panicle_images/'  #训练图文件夹
    SAVE_APTH = './'  #训练集保存路径
    生成两个npy文件，分别为图像和标注
    最新穗株图像百度云链接：链接：https://pan.baidu.com/s/1tu4j5DRlosDcDWh3HQAT0A 密码：zy0i
### 2.运行train_script.py进行训练，需要定义的参数为：
    NET_INPUT_SIZE=80 #网络输入尺寸，只能选择48,80,112,160四种之一
    batch_size=16
    epoch=10
    以及训练集路径
    X = np.load('F:/代码临时/train_images.npy')
    Y = np.load('F:/代码临时/train_groundtruth.npy')
    训练结束后，得到相应的checkpoint文件夹
### 3.运行checkpoint_to_android.py来转换checkpoint文件，生成pb文件
    然后移植到安卓设备
    需要定义的参数为：
    SAVE_PATH = './' #自行设置保存位置
    NET_INPUT_SIZE = 80  #网络输入尺寸


