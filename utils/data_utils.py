import random
import numpy as np
import cv2
from torchio.transforms import (
    RandomNoise,
    RandomMotion,
    Compose,
    RandomBlur
)


def random_crop_3d(img, label, crop_size):
    """
    随机裁剪3d图像为指定尺寸
    :param img: 需要裁减的3d图像
    :param label: 需要裁减的3d label数据
    :param crop_size: 裁减尺寸(x,y,z)
    :return:裁减之后的图像以及label数据，若不符合条件则返回-1
    """
    random_x_max = img.shape[0] - crop_size[0]
    random_y_max = img.shape[1] - crop_size[1]
    random_z_max = img.shape[2] - crop_size[2]
    if random_x_max < 0 or random_y_max < 0 or random_z_max < 0:
        return -1, -1

    x_random = random.randint(0, random_x_max)
    y_random = random.randint(0, random_y_max)
    z_random = random.randint(0, random_z_max)
    crop_img = img[x_random:x_random + crop_size[0],
                   y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
    crop_label = label[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                       z_random:z_random + crop_size[2]]
    return crop_img, crop_label


def random_crop_of_3d(img, dim: int, size: int):
    """
    裁减3d数据中的某一维度
    :param img: 需要裁减的3D图像
    :param dim:裁减的维度
    :param size:需要裁减到什么尺寸
    :return:
    """
    random_max = img.shape[dim] - size
    if random_max < 0:
        return None
    start = random.randint(0, random_max)

    if dim == 0:
        crop_img = img[start:start + size]
    elif dim == 1:
        crop_img = img[:, start:start + size, :]
    else:
        crop_img = img[:, :, start:start + size]
    return crop_img


def resize(img, size, interpolation=cv2.INTER_LINEAR):
    """
    resize 3d图像数据成指定形状
    :param img: 需要处理的图像
    :param size: 变形后的尺寸
    :param interpolation: 差值方式，默认使用线性差值
    :return: resize之后的图像数据
    """
    new_img_np = np.zeros((img.shape[0], size[1], size[0]))
    for i in range(0, img.shape[0]):
        new_img_np[i, :, :] = cv2.resize(
            img[i, :, :], size, interpolation=interpolation)
    return new_img_np


def resize_zy(img, size, interpolation=cv2.INTER_LINEAR):
    """
    从z方向和y方向resize 3d图像数据成指定形状
    :param img: 需要处理的图像
    :param size: 变形后的尺寸
    :param interpolation:interpolation: 差值方式，默认使用线性差值
    :return:resize之后的图像数据
    """
    new_img_np = np.zeros((size[1], size[0], img.shape[1]))
    for i in range(0, img.shape[2]):
        new_img_np[:, :, i] = cv2.resize(
            img[:, :, i], size, interpolation=interpolation)
    return new_img_np


def flip_image_3d(img, label, probability=0.5):
    """
    随机翻转图像
    :param img: 需要处理的图像
    :param label: 需要处理的label数据
    :param probability:有多少概率在维度上翻转，默认0.5
    :return: 处理后的图像和label
    """
    if random.random() > probability:
        img, label = img[::-1, :, :], label[::-1, :, :]
    if random.random() > probability:
        img, label = img[:, ::-1, :], label[:, ::-1, :]
    if random.random() > probability:
        img, label = img[:, :, ::-1], label[:, :, ::-1]
    return img, label


def brightness_contrast(img, probability=0.5):
    """
    随机调整亮度和对比度
    :param img:需要处理的图像
    :param probability:调整的概率值，默认0.5
    :return:处理后的图像
    """
    if random.random() > probability:
        # 调整对比度和亮度
        alpha = random.random() * 3  # 对比度 0-3
        beta = random.random() * 100  # 亮度 范围0-100
        conv_img = alpha * img + beta
        return conv_img
    return img


def transforms_3d(img):
    """
    3d  transform
    :param img: 需要处理的图像
    :return: 处理之后的图像
    """
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    trans = Compose([
        RandomMotion(),  # 添加运动伪影
        RandomBlur(),  # 模糊影像
        RandomNoise(std=(0, 0.5))  # 随机噪音
    ])
    img = trans(img)
    img = img.squeeze(0)

    return img
