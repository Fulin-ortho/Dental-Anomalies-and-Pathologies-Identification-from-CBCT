import SimpleITK as sitk
from utils.cbct_utils import resize_image_itk
import numpy as np
import torch
from net.net3d import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seg_net_1 = UNet(1, 1, 16).to(device).eval()
seg_net_1.load_state_dict(torch.load(
    r'./your_weights', map_location=device, weights_only=True))

seg_net_2 = UNet(2, 1, 16).to(device).eval()
seg_net_2.load_state_dict(torch.load(
    r'./your_weights', map_location=device, weights_only=True))


def run_two_seg_1(cbct_data, box):
    """
    整副牙齿二分类粗分网络
    :params cbct_data:读取的cbct numpy数据
    :params box:整副牙齿的包围盒区域 
    :return:切片之后的数据，网络输出的二分类数据，以及切片之后的形状
    """

    cutting_data_np = cbct_data[box[0]:box[1],
                                box[2]: box[3], box[4]: box[5]]  # 将数组进行切片
    z, y, x = cutting_data_np.shape  # 切片之后的形状数据

    sitk_data = sitk.GetImageFromArray(cutting_data_np.astype(np.int16))
    ori_data_np = resize_image_itk(
        # 进行resize，送入网络分析的数据
        sitk_data, (128, 128, 128), True, sitk.sitkLinear)

    ori_data_np = ori_data_np.reshape(1, 1, 128, 128, 128)
    # 将数据置为[0,1]之间
    ori_data_np[ori_data_np < 0] = 0
    ma = np.max(ori_data_np)
    ori_data_np = ori_data_np / ma

    data = torch.from_numpy(ori_data_np).type(torch.FloatTensor).to(device)
    out = torch.sigmoid(seg_net_1(data))
    out = out.squeeze().cpu().data.numpy()
    out[out < 0.5] = 0
    out[out >= 0.5] = 1
    sitk_out = sitk.GetImageFromArray(out.astype(np.uint8))

    out_open = sitk.BinaryMorphologicalOpening(sitk_out != 0)  # 经过一次开运算
    out_np = resize_image_itk(out_open, (x, y, z), True)  # 形状还原
    return cutting_data_np, out_np, (z, y, x)


def run_two_seg_2(net1_np, img):
    """
    二分类精细分割网络
    :params net1_np: 网络一分割的结果，numpy数据，只包含整副牙齿
    :params img:原始图像数据，numpy数据 
    :return:精细切割二分类结果
    """

    deep, size = 128, 128  # 裁剪的尺寸
    two = net1_np
    z, y, x = two.shape

    data_list_obj = []
    z_num, y_num, x_num = z // deep, y // size, x // size  # 分别计算z,y,x方向上能完整滑动多少次
    for z_i in range(0, z_num + 1):
        for y_i in range(0, y_num + 1):
            for x_i in range(0, x_num + 1):
                z_parm, y_parm, x_parm = [z_i * deep, z_i * deep + deep], [y_i * size, y_i * size + size], [x_i * size,
                                                                                                            x_i * size + size]
                if x_i == x_num:
                    x_parm = [-size, x + 1]
                if y_i == y_num:
                    y_parm = [-size, y + 1]
                if z_i == z_num:
                    z_parm = [-deep, z + 1]
                cutting_np = img[z_parm[0]: z_parm[1],
                                 y_parm[0]:y_parm[1], x_parm[0]:x_parm[1]]
                cutting_np_two = two[z_parm[0]: z_parm[1],
                                     y_parm[0]:y_parm[1], x_parm[0]:x_parm[1]]
                data_list_obj.append({'data': cutting_np, 'position': [z_parm, y_parm, x_parm],
                                      'two_seg': cutting_np_two})
    nii_res = np.zeros(two.shape)  # 创建一个和原始数组一样的空数组

    # 开始分析
    for _, obj in enumerate(data_list_obj):
        data = obj['data']
        two_seg = obj['two_seg']
        position = obj['position']
        s = data.shape
        # 归一化data
        data[data < 0] = 0
        ma = np.max(data)
        data = data / ma
        # 加一个判断尺寸的操作，防止切片范围太小影响reshape
        if s[0] != deep or s[1] != size or s[2] != size:

            data = resize_image_itk(sitk.GetImageFromArray(data.astype(
                np.int16)), (size, size, deep), True, sitk.sitkLinear)
            two_seg = resize_image_itk(sitk.GetImageFromArray(
                two_seg.astype(np.uint8)), (size, size, deep), True)

        data = data.reshape(1, 1, deep, size, size)
        two_seg = two_seg.reshape(1, 1, deep, size, size)
        data = torch.from_numpy(data).type(torch.FloatTensor).to(device)

        two_seg = torch.from_numpy(two_seg).type(torch.FloatTensor).to(device)
        input_data = torch.cat((data, two_seg), dim=1)  # 输入网络数据
        out = torch.sigmoid(seg_net_2(input_data))

        out = out.squeeze().cpu().data.numpy()
        out[out < 0.5] = 0
        out[out >= 0.5] = 1
        # 这里需要判断一下输入形状，如果经过了resize这里需要还原
        if s[0] != deep or s[1] != size or s[2] != size:
            out = resize_image_itk(sitk.GetImageFromArray(out.astype(
                np.uint8)), (s[2], s[1], s[0]), True)

        nii_res[position[0][0]:position[0][1], position[1][0]:position[1][1], position[2][0]: position[2][1]] = out

    return nii_res


def start_two_seg(cbct_data, box):
    """
    牙齿二分类
    :params cbct_data:读取的cbct numpy数据
    :params box:整副牙齿的包围盒区域
    :return:切片之后的数据，整口牙精细二分类的数据，以及切片之后的形状
    """
    cutting_data_np, out_np, (z, y, x) = run_two_seg_1(cbct_data, box)
    seg_out = run_two_seg_2(out_np, cutting_data_np)
    return cutting_data_np, seg_out, (z, y, x)
