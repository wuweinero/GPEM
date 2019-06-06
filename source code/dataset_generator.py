# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:11:53 2017

@author: minus
"""

import cv2, os, sys, math
import random
import numpy as np

debug = False


def rotate(
        img,  #image matrix  
        angle=0,  #angle of rotation,
        scale=1):

    height = img.shape[0]
    width = img.shape[1]

    rotateMat = cv2.getRotationMatrix2D((width // 2, height // 2), angle,
                                        scale)
    rotateImg = cv2.warpAffine(
        img, rotateMat, (width, height), borderMode=cv2.BORDER_REPLICATE)

    return rotateImg  #rotated imag


def square_boder(img, btype=cv2.BORDER_WRAP):
    diff = img.shape[0] - img.shape[1]
    if abs(diff) < 2:
        return img.copy()
    elif diff < -1:

        diff = abs(diff)
        return cv2.copyMakeBorder(img, diff // 2, diff - diff // 2, 0, 0,
                                  btype)
    else:

        diff = abs(diff)
        return cv2.copyMakeBorder(img, 0, 0, diff // 2, diff - diff // 2,
                                  btype)


def random_tansform(img_list, angle_range=[0, 360], scale_range=[0.5, 1.1]):
    angle = random.random() * (
        angle_range[0] - angle_range[1]) + angle_range[1]
    scale = random.random() * (
        scale_range[0] - scale_range[1]) + scale_range[1]
    ret_imgs = []
    for img in img_list:
        img_tem = square_boder(img, cv2.BORDER_REPLICATE)
        ret_imgs.append(rotate(img_tem, angle, scale))

    return ret_imgs


def extract_label(mask_png):
    mask = np.zeros_like(mask_png)
    is_R = mask_png[:, :, 0] == 255
    is_G = mask_png[:, :, 1] == 255
    is_B = mask_png[:, :, 2] == 255
    no_R = mask_png[:, :, 0] == 0
    no_G = mask_png[:, :, 1] == 0
    no_B = mask_png[:, :, 2] == 0

    mask[:, :, 0] = is_R * no_B * no_G
    mask[:, :, 1] = is_G * no_R * no_B
    mask[:, :, 2] = is_B * no_R * no_G

    mask *= 255
    return mask


def generate_tarinset(ImgList,
                      MaskList,
                      epoch=5,
                      seed_num=1000,
                      cond1_weight=0.5):
    '''
    穗型网络的训练集生成函数
    Args:
        ImgList: 原图文件路径列表
        MaskList: 标注图文件路径列表
        epoch: 循环次数
        seed_num: 每张图随机试探的次数
        cond1_weight: 正样本占的概率权重
    Returns:
        train_images: shape=(N,160,160,3)的样本集
        train_groundtruth: shape=(N,160,160,1)的标注集
    '''
    sample_size = [160, 160]

    train_groundtruth = []
    train_images = []

    total_number_of_images = len(ImgList)

    number_cond_1 = 0
    number_cond_2 = 0

    for cycle in range(epoch):
        for index in range(total_number_of_images):

            RGB = cv2.imread(ImgList[index])[:, :, ::-1]  #RGB
            mask = cv2.imread(MaskList[index])[:, :, ::-1]  #RGB
            mask = extract_label(mask)

            RGB = cv2.resize(
                RGB, (0, 0), fx=0.77, fy=0.77, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(
                mask, (0, 0), fx=0.77, fy=0.77, interpolation=cv2.INTER_AREA)
            mask //= 255

            if debug:
                cv2.imshow('rgb', RGB)
                cv2.imshow('mask', mask * 255)
                cv2.waitKey(10000)

            r_im = random_tansform([RGB, mask], [-180, 180], [0.8, 1.1])

            number = 0

            gray = cv2.cvtColor(r_im[0], cv2.COLOR_RGB2HSV)
            gray = gray[:, :, 1] > 100

            for i in range(seed_num):
                x_s = random.randint(0, r_im[0].shape[1] - sample_size[1])
                y_s = random.randint(0, r_im[0].shape[0] - sample_size[0])

                im_crop_mask = r_im[1][y_s:y_s + sample_size[0], x_s:
                                       x_s + sample_size[1], :]
                im_crop_rgb = r_im[0][y_s:y_s + sample_size[0], x_s:
                                      x_s + sample_size[1], :]
                im_crop_gray = gray[y_s:y_s + sample_size[0], x_s:
                                    x_s + sample_size[1]]

                conditon_1 = (im_crop_mask[30:-30, 30:-30, 1].sum() >
                              40) and (random.random() < cond1_weight)
                conditon_2 = (im_crop_mask[30:-30, 30:-30, 0].sum() +
                              im_crop_mask[30:-30, 30:-30, 2].sum()) > 2
                number_cond_1 += conditon_1
                number_cond_2 += conditon_2
                if conditon_1 or conditon_2:
                    mask_tem = np.zeros_like(im_crop_mask[:, :, 1])
                    _, cnts, _ = cv2.findContours(
                        im_crop_mask[:, :, 1].copy() +
                        im_crop_mask[:, :, 2].copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_NONE)
                    for cnt in cnts:
                        x1, y1, w1, h1 = cv2.boundingRect(cnt)

                        cv2.circle(mask_tem, (x1 + w1 // 2, y1 + h1 // 2), 8,
                                   1, -1)

                    if random.random() > 0.5:
                        mask_tem = mask_tem[::-1, :]
                        im_crop_rgb = im_crop_rgb[::-1, :, :]
                        im_crop_gray = im_crop_gray[::-1, :]

                    if random.random() > 0.5:
                        mask_tem = mask_tem[:, ::-1]
                        im_crop_rgb = im_crop_rgb[:, ::-1, :]
                        im_crop_gray = im_crop_gray[:, ::-1]

                    train_groundtruth.append(
                        mask_tem.reshape(mask_tem.shape + (1, )))
                    train_images.append(im_crop_rgb)

                    number += 1


#                    plt.imshow(im_crop_gray*255)
#                    plt.show()
#                    plt.imshow(im_crop_rgb)
#                    plt.show()
#                    plt.imshow(mask_tem*255)
#                    plt.show()

            print('epoch', cycle, 'img index', index, 'croped batch number',
                  number)
            if debug:
                cv2.imshow('rgb2', r_im[0])
                cv2.imshow('mask', r_im[1])
                cv2.waitKey(500)
    if debug:
        cv2.destroyAllWindows()

    print('toatl szie:%d,cond1:%d,cond2:%d' % (len(train_images),
                                               number_cond_1, number_cond_2))
    train_groundtruth = np.array(train_groundtruth)
    train_images = np.array(train_images)

    return train_images, train_groundtruth

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    IMG_PATH = 'F:/panicle_images/'  #训练图文件夹
    SAVE_APTH = './'  #训练集保存路径
    AllImgList = [
        os.path.join(IMG_PATH, x) for x in os.listdir(IMG_PATH)
        if x[-3:] == 'jpg'
    ]
    AllImgList.sort()

    ImgList, MaskList = [], []

    for p in AllImgList:
        if os.path.exists(p[:-4] + '_mask.png'):
            MaskList.append(p[:-4] + '_mask.png')
            ImgList.append(p)

    print('采集图像总个数：', len(AllImgList))
    print('有标注图像个数：', len(ImgList))

    train_images, train_groundtruth = generate_tarinset(
        ImgList, MaskList, epoch=2, seed_num=500)

    print('saving...')
    np.save(
        os.path.join(SAVE_APTH, 'train_groundtruth.npy'), train_groundtruth)
    np.save(os.path.join(SAVE_APTH, 'train_images.npy'), train_images)
    print('trainset has been saved')
