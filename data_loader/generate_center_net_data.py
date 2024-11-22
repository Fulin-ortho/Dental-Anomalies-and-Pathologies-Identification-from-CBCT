
import numpy as np
import SimpleITK as sitk
import glob
import os
from utils.cbct_utils import draw_umich_gaussian, gaussian_radius
import random


def resize_image_itk(itkimage, new_size, np_array=False, resamplemethod=sitk.sitkNearestNeighbor):
    """
    将sitk图像resize成新的形状
    :params itkimage:sitk图像 
    :params new_size:需要resize的形状 
    :params np_array:是否将转换形状之后的数据转换为numpy数据
    :params resamplemethod:插值方式
    :return:变换形状之后的结果
    """
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(new_size, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int32)  # spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    if np_array:
        return sitk.GetArrayFromImage(itkimgResampled)
    else:
        return itkimgResampled


def sitk_read_raw(img_path):
    """
    读取3D图像并resale（因为一般医学图像并不是标准的[1,1,1]scale）
    :params img_path: 图像路径
    :params resize_scale: resize的尺度
    :return:numpy的3D图像数据
    """
    nda = sitk.ReadImage(img_path)
    if nda is None:
        raise TypeError("input img is None!!!")
    nda = sitk.GetArrayFromImage(nda)  # channel first

    return nda


def generate_data(label_path, txt_path, save_path):
    """
    :param label_path: 单颗牙齿分割标签
    :param txt_path:牙齿标签信息
    :param save_path:保存路径
    :return:
    """

    SIZE = 128
    start_size = (128, 128, 128)  # 输入形状

    down_ratio = 128/SIZE  # 下采样因子
    label_list = glob.glob(label_path)
    txt_list = glob.glob(txt_path)

    save_npy = os.path.join(save_path, 'npy')
    save_data = os.path.join(save_path, 'data')
    if not os.path.exists(save_npy):
        os.makedirs(save_npy)
        os.makedirs(save_data)
    teeth_list = ['11"', '12"', '13"', '21"', '22"', '23"', '31"', '32"', '33"', '41"', '42"', '43"', '14"', '15"',
                  '24"', '25"', '34"', '35"', '44"', '45"', '16"', '17"', '18"', '26"', '27"', '28"', '36"', '37"',
                  '38"', '46"', '47"', '48"']  # 牙齿标记
    for idx, l_path in enumerate(label_list):
        print(l_path, idx)
        heatmap = np.zeros((SIZE, SIZE, SIZE), dtype=np.float32)  # 热图
        whd_target = np.zeros((3, SIZE, SIZE, SIZE), dtype=np.float32)
        whd_mask = np.zeros((3, SIZE, SIZE, SIZE), dtype=np.float32)
        center_reg = np.zeros((3, SIZE, SIZE, SIZE), dtype=np.float32)
        center_reg_mask = np.zeros((3, SIZE, SIZE, SIZE), dtype=np.float32)
        file, _ = os.path.splitext(l_path)
        _, name = os.path.split(file)
        npy_name, _ = os.path.splitext(name)
        # 读取标签内容
        with open(txt_list[idx]) as f:
            content = f.readlines()
        content = content[15:]  # 去掉头部信息
        sitk_img = sitk.ReadImage(l_path)
        # 将数据resize成输入形状
        teeth_np = resize_image_itk(sitk_img, start_size, True)

        # 解析读取的标签信息
        for la in content:
            split = la.split()
            if split[-1] in teeth_list:  # 判断牙齿的label是否在需要识别的牙齿数据中

                color_num = int(split[0])  # 获取牙齿编号
                # 获取牙齿边框
                z, y, x = np.where(teeth_np == color_num)
                # 排除没有删除的编号数据
                if len(x) == 0 or len(y) == 0 or len(z) == 0:
                    continue
                # 获取牙齿边界
                z_min, z_max = np.min(z), np.max(z)
                y_min, y_max = np.min(y), np.max(y)
                x_min, x_max = np.min(x), np.max(x)
                # 获取目标在整张图中的宽度、高度和深度
                w, h, d = x_max-x_min, y_max-y_min, z_max-z_min

                # w, h, d = w/SIZE, h/SIZE, d/SIZE  # 归一化
                # 计算目标的中心点
                center = np.array(
                    [(z_min+z_max)/2/down_ratio, (y_min+y_max)/2/down_ratio, (x_min+x_max)/2/down_ratio], dtype=np.float32)
                center_int = center.astype(np.int32)  # 取整之后的中心点

                cz, cy, cx = center_int
                # 根据牙齿的d w h 获取半径
                radius = gaussian_radius(
                    (np.ceil(d/down_ratio), np.ceil(h/down_ratio), np.ceil(w/down_ratio)), 0.6)

                radius = int(round(radius))  # 保证每一个中心点都能被高斯模糊
                if radius == 0:
                    radius = 1  # 设置最小半径为1

                # radius = 3  # 半径给固定值，防止半径太小，生成的高斯核太小，影响精度
                heatmap = draw_umich_gaussian(
                    heatmap, center_int, int(radius))

                # 生成宽高标签
                whd_target[0, cz, cy, cx] = d/down_ratio
                whd_target[1, cz, cy, cx] = h/down_ratio
                whd_target[2, cz, cy, cx] = w/down_ratio
                whd_mask[:, cz, cy, cx] = 1.0

                # 计算中心点偏移量，浮点数相对于整数坐标的偏移量
                center_reg[:, cz, cy, cx] = center - center_int

                center_reg_mask[:, cz, cy, cx] = 1.0

        res_label = np.concatenate(
            (np.expand_dims(heatmap, 0), whd_target, whd_mask, center_reg, center_reg_mask))

        # 保存标签数据
        np.savez_compressed(os.path.join(
            save_npy, npy_name+'.npz'), res_label, data=res_label)
        # 将牙齿数据二值化之后，保存
        teeth_np[teeth_np != 0] = 1
        teeth_img = sitk.GetImageFromArray(teeth_np.astype(np.uint8))
        sitk.WriteImage(teeth_img, os.path.join(save_data, name+'.gz'))


def generate_data_defect(label_path, txt_path, save_path, defect_num=1):
    SIZE = 128
    start_size = (128, 128, 128)  # 需要变换的形状
    down_ratio = 128/SIZE  # 下采样因子
    label_list = glob.glob(label_path)
    txt_list = glob.glob(txt_path)

    save_npy = os.path.join(save_path, 'npy')
    save_data = os.path.join(save_path, 'data')
    if not os.path.exists(save_npy):
        os.makedirs(save_npy)
        os.makedirs(save_data)
    teeth_list = ['11"', '12"', '13"', '21"', '22"', '23"', '31"', '32"', '33"', '41"', '42"', '43"', '14"', '15"',
                  '24"', '25"', '34"', '35"', '44"', '45"', '16"', '17"', '18"', '26"', '27"', '28"', '36"', '37"',
                  '38"', '46"', '47"', '48"']  # 牙齿标记
    for idx, l_path in enumerate(label_list):
        defect_list = []  # 缺牙齿的数组
        # 随机缺牙齿,缺几颗牙齿
        num = random.randint(1, defect_num)
        while len(defect_list) < num:
            defect = random.choice(teeth_list)
            if defect not in defect_list:
                defect_list.append(defect)

        heatmap = np.zeros((SIZE, SIZE, SIZE), dtype=np.float32)  # 热图
        whd_target = np.zeros((3, SIZE, SIZE, SIZE), dtype=np.float32)
        whd_mask = np.zeros((3, SIZE, SIZE, SIZE), dtype=np.float32)
        center_reg = np.zeros((3, SIZE, SIZE, SIZE), dtype=np.float32)
        center_reg_mask = np.zeros((3, SIZE, SIZE, SIZE), dtype=np.float32)
        file, _ = os.path.splitext(l_path)
        _, name = os.path.split(file)
        npy_name, _ = os.path.splitext(name)
        # 读取标签内容
        with open(txt_list[idx]) as f:
            content = f.readlines()
        content = content[15:]  # 去掉头部信息
        sitk_img = sitk.ReadImage(l_path)
        # 将数据resize成新的形状
        teeth_np = resize_image_itk(sitk_img, start_size, True)

        # 解析读取的标签信息
        for la in content:
            split = la.split()
            if split[-1] in teeth_list:  # 判断牙齿的label是否在需要识别的牙齿数据中
                color_num = int(split[0])  # 获取牙齿编号
                if split[-1] in defect_list:  # 准备缺的牙齿
                    mask = teeth_np == color_num
                    teeth_np[mask] = 0  # 将这颗牙齿值变为0
                # 获取牙齿边框
                z, y, x = np.where(teeth_np == color_num)
                # 排除没有删除的编号数据
                if len(x) == 0 or len(y) == 0 or len(z) == 0:
                    continue
                # 获取牙齿边界
                z_min, z_max = np.min(z), np.max(z)
                y_min, y_max = np.min(y), np.max(y)
                x_min, x_max = np.min(x), np.max(x)
                # 获取目标在整张图中的宽度、高度和深度
                w, h, d = x_max-x_min, y_max-y_min, z_max-z_min

                # w, h, d = w/SIZE, h/SIZE, d/SIZE  # 归一化
                # 计算目标的中心点
                center = np.array(
                    [(z_min+z_max)/2/down_ratio, (y_min+y_max)/2/down_ratio, (x_min+x_max)/2/down_ratio], dtype=np.float32)
                center_int = center.astype(np.int32)  # 取整之后的中心点
                cz, cy, cx = center_int
                # 根据牙齿的d w h 获取半径
                radius = gaussian_radius(
                    (np.ceil(d/down_ratio), np.ceil(h/down_ratio), np.ceil(w/down_ratio)), 0.6)

                radius = int(round(radius))  # 保证每一个中心点都能被高斯模糊
                if radius == 0:
                    radius = 1  # 设置最小半径为1

                # radius = 3  # 半径给固定值，防止半径太小，生成的高斯核太小，影响精度
                heatmap = draw_umich_gaussian(
                    heatmap, center_int, int(radius))
                # 生成宽高标签
                whd_target[0, cz, cy, cx] = d/down_ratio
                whd_target[1, cz, cy, cx] = h/down_ratio
                whd_target[2, cz, cy, cx] = w/down_ratio
                whd_mask[:, cz, cy, cx] = 1.0

                # 计算中心点偏移量，浮点数相对于整数坐标的偏移量
                center_reg[:, cz, cy, cx] = center - center_int

                center_reg_mask[:, cz, cy, cx] = 1.0
        res_label = np.concatenate(
            (np.expand_dims(heatmap, 0), whd_target, whd_mask, center_reg, center_reg_mask))
        # 保存标签数据
        np.savez_compressed(os.path.join(
            save_npy, npy_name+'.npz'), res_label, data=res_label)
        # 将牙齿数据二值化之后，保存
        teeth_np[teeth_np != 0] = 1
        teeth_img = sitk.GetImageFromArray(teeth_np.astype(np.uint8))
        sitk.WriteImage(teeth_img, os.path.join(save_data, name+'.gz'))


if __name__ == '__main__':

    generate_data(r'your_path\label\*',
                  r'your_path\txt\*', r'your_path\val')
