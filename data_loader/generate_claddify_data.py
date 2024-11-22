import numpy as np
import SimpleITK as sitk
import glob
import os
import random
from utils.cbct_utils import sitk_read_raw, resize_image_itk


def generate_classify_teeth(label_path, txt_path, save_path, range: int = 1):
    """
    生成单颗牙齿分类数据
    :params : 
    :return:
    """
    save_features = r'your_path'
    label_list = glob.glob(label_path)
    txt_list = glob.glob(txt_path)
    teeth_list = ['11"', '12"', '13"', '14"', '15"', '16"', '17"', '18"', '21"', '22"', '23"', '24"', '25"', '26"',
                  '27"', '28"', '31"', '32"', '33"', '34"', '35"', '36"', '37"', '38"', '41"', '42"', '43"', '44"',
                  '45"', '46"', '47"', '48"']  # 需要处理的牙齿编号

    for index, path in enumerate(label_list):
        file, _ = os.path.splitext(txt_list[index])
        _, name = os.path.split(file)
        print(name, index)
        label_np = sitk_read_raw(path)
        z_, y_, x_ = label_np.shape  # label形状
        with open(txt_list[index]) as f:
            content = f.readlines()
        label_data = content[15:]
        for la in label_data:
            split = la.split()
            if split[-1] in teeth_list:
                color_num = float(split[0])  # 牙齿颜色编号
                z, y, x = np.where(label_np == color_num)
                if len(z) <= 5 or len(y) <= 5 or len(x) <= 5:
                    continue
                z_min, z_max = np.min(z), np.max(z)
                y_min, y_max = np.min(y), np.max(y)
                x_min, x_max = np.min(x), np.max(x)
                s = np.array([(z_max-z_min)/z_, (y_max-y_min)/y_,
                              (x_max-x_min)/x_], dtype=np.float32)  # 牙齿长宽高,并且归一化
                center = np.array(
                    # 中心点
                    [(z_min+z_max)/2/x_, (y_min+y_max)/2/y_, (x_min+x_max)/2/z_], dtype=np.float32)

                features = np.hstack((center, s))

                # 重新定义range
                if range == 0:
                    deep, height, width = z_max - z_min, y_max - y_min, x_max - x_min
                    z_mi, z_ma, y_mi, y_ma, x_mi, x_ma = deep // 4, deep // 4, height // 4, height // 4, width // 4, width // 4
                else:
                    z_mi, z_ma, y_mi, y_ma, x_mi, x_ma = random.randint(1, range), random.randint(1,
                                                                                                  range), random.randint(
                        1,
                        range), random.randint(
                        1, range), random.randint(1, range), random.randint(1, range)
                # 如果超出了边界范围则取边界值
                if z_min - z_mi < 0:
                    z_mi = z_min
                if z_max + z_ma > z_:
                    z_ma = z_ - z_max
                if y_min - y_mi < 0:
                    y_mi = y_min
                if y_max + y_ma > y_:
                    y_ma = y_ - y_max
                if x_min - x_mi < 0:
                    x_mi = x_min
                if x_max + x_ma > x_:
                    x_ma = x_ - x_max
                # 将牙齿切片
                label = label_np[z_min - z_mi:z_max + z_ma, y_min -
                                 y_mi:y_max + y_ma, x_min - x_mi:x_max + x_ma]
                label_zero = np.zeros(label.shape)
                label_zero[label == color_num] = 1  # 生成label

                label_zero = sitk.GetImageFromArray(label_zero.astype('uint8'))
                label_zero = resize_image_itk(label_zero, (64, 64, 96))
                sitk.WriteImage(label_zero,
                                os.path.join(save_path, name + '_' + split[-1].split('"')[0] + '.nii.gz'))
                np.save(os.path.join(save_features, name +
                                     '_' + split[-1].split('"')[0]), features)
