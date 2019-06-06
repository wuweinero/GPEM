# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 20:52:48 2017

@author: minus
穗型谷粒数检测网络的类，包含穗型网络结构，训练和预测的函数
"""

import time
from tqdm import tqdm  #进度条模块
import os
# which gpu you train, set id to none 0 if want to train on cpu
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
tf.reset_default_graph()


def conv_conv_pool(input_,
                   n_filters,
                   training,
                   name,
                   pool=True,
                   activation=tf.nn.relu):

    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(
                net,
                F, (3, 3),
                activation=activation,
                padding='same',
                name="conv_{}".format(i + 1))

        if pool is False:
            print(net)
            return net

        pool = tf.layers.max_pooling2d(
            net, (2, 2), strides=(2, 2), name="pool_{}".format(name))
        print(pool)
        return net, pool


def upsample_concat(inputA, input_B, name):

    upsample = upsampling_2D(inputA, size=(2, 2), name=name)

    return tf.concat(
        [upsample, input_B], axis=-1, name="concat_{}".format(name))


def upsampling_2D(tensor, name, size=(2, 2)):

    H, W, _ = tensor.get_shape().as_list()[1:]

    H_multi, W_multi = size
    target_H = H * H_multi
    target_W = W * W_multi

    return tf.image.resize_nearest_neighbor(
        tensor, (target_H, target_W), name="upsample_{}".format(name))


def make_unet(X, training, scale=12):

    net = X / 127.5 - 1  #RGB格式图像归一化图像
    net = tf.layers.conv2d(net, 3, (1, 1), name="color_space_adjust")

    conv1, pool1 = conv_conv_pool(
        net, [1 * scale, 1 * scale], training, name=1)
    conv2, pool2 = conv_conv_pool(
        pool1, [2 * scale, 2 * scale], training, name=2)
    conv3, pool3 = conv_conv_pool(
        pool2, [4 * scale, 4 * scale], training, name=3)
    conv4, pool4 = conv_conv_pool(
        pool3, [8 * scale, 8 * scale], training, name=4)
    conv5 = conv_conv_pool(
        pool4, [16 * scale, 16 * scale], training, name=5, pool=False)

    up6 = upsample_concat(conv5, conv4, name=6)
    conv6 = conv_conv_pool(
        up6, [8 * scale, 8 * scale], training, name=6, pool=False)

    up7 = upsample_concat(conv6, conv3, name=7)
    conv7 = conv_conv_pool(
        up7, [4 * scale, 4 * scale], training, name=7, pool=False)

    up8 = upsample_concat(conv7, conv2, name=8)
    conv8 = conv_conv_pool(
        up8, [2 * scale, 2 * scale], training, name=8, pool=False)

    up9 = upsample_concat(conv8, conv1, name=9)
    conv9 = conv_conv_pool(
        up9, [1 * scale, 1 * scale], training, name=9, pool=False)

    conv10 = tf.layers.conv2d(conv9, 1, (1, 1), name='conv10', padding='same')

    output = tf.nn.sigmoid(conv10, name='output')

    return output


def predict(imgs, sess, global_X, global_pred):

    batch_size = 8

    pred_result = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2], 1),
                           np.float32)

    bar = tqdm(range(0, imgs.shape[0], batch_size))  #进度条定义
    bar.set_description('谷粒数检测进度')
    for i in bar:
        output = sess.run(
            [global_pred], feed_dict={global_X: imgs[i:i + batch_size]})
        pred_result[i:i + batch_size] = np.asarray(output)

    if (i + batch_size < imgs.shape[0]):
        output = sess.run(
            [global_pred],
            feed_dict={global_X: imgs[i + batch_size:imgs.shape[0]]})
        pred_result[i + batch_size:imgs.shape[0]] = np.asarray(output)

    return pred_result


def predict_one(imgs, sess, global_X, global_pred):

    pred_result = np.zeros((1, imgs.shape[0], imgs.shape[1], 1), np.float32)
    output = sess.run(
        [global_pred], feed_dict={global_X: imgs.reshape((1, ) + imgs.shape)})
    pred_result = np.asarray(output)

    return pred_result[0][0, :, :, 0]


def predict_full_image(img, sess, global_X, global_pred, ISZ):
    '''
    预测整张图片的谷粒位置；
    img：RGB格式图像，0-255；
    sess: tensorflow会话，包含预测模型；
    global_X:输入tensor；
    global_pred:输出tensor；
    ISZ:网络模型的输入尺寸
    '''
    width, height = img.shape[1], img.shape[0]
    if img.dtype == np.uint8:
        img = img.astype(np.float32)

    overlap_size = 6  #裁剪后的子图之间的重叠宽度
    crop_size = 6  #从预测输出的图像上裁剪的边界宽度

    pred_size = ISZ - 2 * crop_size  #裁剪后预测图像的大小
    step = pred_size - overlap_size  #预测时，窗口滑动的步长

    m = (width - overlap_size - 1) // (pred_size - overlap_size) + 1  #水平窗口个数
    n = (height - overlap_size - 1) // (pred_size - overlap_size) + 1  #垂直窗口个数

    #扩展后的输出图像大小
    tem_w = crop_size * 2 + (pred_size - overlap_size) * m + overlap_size
    tem_h = crop_size * 2 + (pred_size - overlap_size) * n + overlap_size

    assert tem_w >= width and tem_h >= height

    cnv = np.zeros((tem_h + 2 * crop_size, tem_w + 2 * crop_size, 3)).astype(
        np.float32)
    cnv[crop_size:height + crop_size, crop_size:width + crop_size, :] = img

    prd = np.zeros((tem_h, tem_w)).astype(np.float32)
    weight_map = np.zeros((tem_h, tem_w))  #重叠权重

    sub_imgs = []  #按窗口滑动裁剪的子图序列

    for i in range(0, n):
        for j in range(0, m):
            sub_imgs.append(
                cnv[i * step:i * step + ISZ, j * step:j * step + ISZ, :])

    x = np.array(sub_imgs)

    tmp = predict(x, sess, global_X, global_pred)
    index = 0
    #按照窗口滑动位置对应的粘贴输出结果
    for i in range(0, n):
        for j in range(0, m):
            prd[i * step :i * step + pred_size,j * step :j * step + pred_size] += \
            tmp[index][crop_size:-crop_size,crop_size:-crop_size,0]
            weight_map[i * step:i * step + pred_size, j * step:
                       j * step + pred_size] += 1
            index += 1
    prd = prd[:height, :width]  #裁剪为原输入图像大小
    weight_map = weight_map[:height, :width]

    prd /= weight_map
    #返回预测的置信度图，像素值接近0，表示此处不可能有谷粒，而接近1时，很可能是谷粒中心
    return prd


def batch_resize(imgs, size=160):
    x = [cv2.resize(imgs[i], (size, size)) for i in range(imgs.shape[0])]
    x = np.array(x)
    if imgs.shape[3] == 1:
        x = x.reshape(x.shape + (1, ))
    return x.astype(np.float32)


def IOU_(y_pred, y_true):
    """Returns a (approx) IOU score
    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)
    Returns:
        float: IOU score
    """
    H, W, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(
        pred_flat, axis=1) + tf.reduce_sum(
            true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)


class PanicleNet():
    def __init__(self, input_size, TRAIN_FLAG=False):

        tf.reset_default_graph()
        # 四种不同的输入尺寸，对应四种不同的模型第一层通道数
        _map = {48: 10, 80: 16, 112: 24, 160: 32}
        self.HH = input_size
        self.Scale = _map[self.HH]
        self.global_X = tf.placeholder(
            tf.float32, shape=[None, self.HH, self.HH, 3], name="input")
        self.global_pred = make_unet(self.global_X, TRAIN_FLAG, self.Scale)

        tf.add_to_collection("inputs", self.global_X)
        tf.add_to_collection("outputs", self.global_pred)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        if not TRAIN_FLAG:
            if os.path.exists("new_model_%d_%d" % (self.HH, self.Scale)):
                latest_check_point = tf.train.latest_checkpoint(
                    "new_model_%d_%d" % (self.HH, self.Scale))

                saver.restore(self.sess, latest_check_point)
                print('loading check_point', latest_check_point)
                print()
            else:
                print('no check point is found!')
                print('please check path', "new_model_%d_%d" % (self.HH,
                                                                self.Scale))

        else:
            print('prepare to train net: input size %d, C1 channels: %d ' %
                  (self.HH, self.Scale))

    def predict_output(self, rgb):
        '''
        输出标注图
        Args:
            rgb:RGB格式输入，稻穗图像
        Rerurns:
            pred_mask:打点标记的图像
        '''

        img = cv2.resize(rgb, (int(rgb.shape[1] * 0.77 * self.HH / 160),
                               int(rgb.shape[0] * 0.77 * self.HH / 160)))

        pred_mask = predict_full_image(
            img.astype(np.float32), self.sess, self.global_X, self.global_pred,
            self.HH)

        return pred_mask

    def predict_number(self, rgb, threshold):
        '''
        输出标注图，并预测谷粒数
        Args:
            rgb:RGB格式输入，稻穗图像
            threshold:置信度阈值，高于此阈值认为是谷粒 范围[0,1]
        Rerurns:
            rgb_out:打点标记的图像
            pred_num:预测数量
            result_points:检测到的谷粒中心坐标点     
        '''
        mask = self.predict_output(rgb)
        rgb_out = rgb.copy()

        pred_mask = mask > threshold
        msk, cnts, _ = cv2.findContours(
            pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)

        result_points = []
        ratio = 0.77 * self.HH / 160

        for cnt in cnts:
            if cv2.contourArea(cnt) > 5 * ratio:
                x, y, w, h = cv2.boundingRect(cnt)
                x0 = int(round((x + w / 2) / ratio))
                y0 = int(round((y + h / 2) / ratio))
                result_points.append((x0, y0))
                cv2.circle(rgb_out, (x0, y0), 8, (0, 255, 0), -1)

        pred_num = len(result_points)

        return rgb_out, pred_num, result_points

    def train(self, X, Y, batch_size=16, epochs=8, learning_rate=0.001):

        assert len(X.shape) == len(Y.shape) == 4
        assert X.shape[:3] == Y.shape[:3]

        n_train = X.shape[0]

        x_input = self.global_X
        y_pred = self.global_pred
        y_true = tf.placeholder(
            tf.float32, shape=[None, self.HH, self.HH, 1], name="label")

        loss = -tf.reduce_mean(
            1 *
            (y_true * tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0)) +
             (1 - y_true) * tf.log(tf.clip_by_value(1 - y_pred, 1e-10, 1.0))))

        optim = tf.train.AdamOptimizer(learning_rate)
        train_op = optim.minimize(loss, global_step=tf.train.get_global_step())

        IOU_op = IOU_(y_pred, y_true)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        saver = tf.train.Saver()
        ckdir = "new_model_%d_%d" % (self.HH, self.Scale)

        try:
            os.mkdir(ckdir)
        except Exception as e:
            pass

        try:

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            the_global_step = 0

            for epoch in range(epochs):

                r_permu = np.random.permutation(n_train)

                X_train = X[r_permu]
                Y_train = Y[r_permu]

                time_begin = time.time()
                for step in range(0, n_train - batch_size, batch_size):

                    x_batch = batch_resize(X_train[step:step + batch_size],
                                           self.HH)
                    y_batch = batch_resize(Y_train[step:step + batch_size],
                                           self.HH)

                    _optim, step_iou, step_loss = self.sess.run(
                        [train_op, IOU_op, loss],
                        feed_dict={
                            x_input: x_batch,
                            y_true: y_batch
                        })
                    time_end = time.time()
                    each_cycle = (time_end - time_begin)
                    time_begin = time_end
                    ETA = (n_train - step) / batch_size * each_cycle

                    the_global_step += 1
                    if the_global_step % 10 == 0:
                        print('Epoch :%d' % epoch,
                              'Step :%d' % the_global_step,
                              "Step_iou :%.3f" % step_iou, "ETA :%d s" % ETA,
                              "loss", step_loss)
                        #TODO record data into excel

                    if (the_global_step % 100 == 0):
                        #TODO testing process
                        print('saving……')
                        saver.save(self.sess, "{}/model.ckpt".format(ckdir))

        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(self.sess, "{}/model.ckpt".format(ckdir))


if __name__ == '__main__':

    pNet = PanicleNet(48, TRAIN_FLAG=True)
    X = np.load('F:/代码临时/train_images.npy')
    Y = np.load('F:/代码临时/train_groundtruth.npy')
    Y = Y.reshape(Y.shape + (1, ))
    print(X.shape, Y.shape)

    pNet.train(X, Y)

    print('training end, testing one image')
    IMG_PATH = 'F:/panicle_images/'
    SAVE_APTH = './'

    AllImgList = [
        os.path.join(IMG_PATH, x) for x in os.listdir(IMG_PATH)
        if x[-3:] == 'jpg'
    ]
    rgb = cv2.imread(AllImgList[0])[:, :, ::-1]
    mask = pNet.predict_output(rgb)
    plt.imshow(mask)
    plt.show()