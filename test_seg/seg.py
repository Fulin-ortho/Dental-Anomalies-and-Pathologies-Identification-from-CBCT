from core.get_2d_box_update import get_teeth_box
from utils.cbct_utils import trans_2d
import SimpleITK as sitk
from core.candidate import detect, generate_slice_teeth
from core.seg_cbct import classify_tooth, combination, start_seg
from core.two_seg_cascade import start_two_seg
import os
import numpy as np
teeth_list = [11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43, 14, 15, 24, 25, 34, 35, 44, 45, 16, 17, 18, 26, 27,
              28, 36, 37, 38, 46, 47, 48]  # 牙齿列表


def seg(cbct_path, save_path, box=[]):
    _, fname = os.path.split(cbct_path)
    teeth_data = {
        'teethBox': [],  # 整副牙齿的包围盒
        'ori_shape': None,  # 原始牙齿形状
        'cutting_data': None,  # 切下来的原始数据,主要会在原始图像上切割会用到
        'two_out': None,  # 二分类牙齿数据
    }  # 牙齿数据
    # 获取orgin和space以便于还原
    sitk_img = sitk.ReadImage(cbct_path)
    origin = sitk_img.GetOrigin()
    space = sitk_img.GetSpacing()
    direction = sitk_img.GetDirection()
    # 获取整副牙齿候选区
    data_np = sitk.GetArrayFromImage(sitk_img)

    if not box:
        z_img, y_img, shape = trans_2d(data_np)
        box = get_teeth_box(z_img, y_img, shape)

    teeth_data['teethBox'] = box
    teeth_data['ori_shape'] = data_np.shape
    cutting_data, two_out, _ = start_two_seg(data_np, box)
    # 保存两个数据
    sitk.WriteImage(sitk.GetImageFromArray(cutting_data.astype(
        'int16')), os.path.join(r'your_path', fname))

    sitk.WriteImage(sitk.GetImageFromArray(two_out.astype(
        'uint8')), os.path.join(r'your_path', fname))

    teeth_data['cutting_data'] = cutting_data
    teeth_data['two_out'] = two_out

    res_teeth_box = detect(two_out, open=False)  # 牙齿候选区数据
    slice_data = generate_slice_teeth(res_teeth_box, two_out)  # 提取单颗牙
    """
       slice_data = [{
        tooth:..., 切片下来的单颗牙齿
        boundary:..., 牙齿边框
        num:..., 编号
        ori_shape_tooth:..., 每一颗牙齿的原始形状
        center:..., 中心点
        }]
       """
    print('开始分割！')
    tooth_list = start_seg(slice_data)
    """
    tooth_list 的长度和slice_data一致
    tooth_list = [{
        tooth:...,分割之后的牙齿，resize之后
        ori_shape_tooth:...,回复到原始形状之后的牙齿
    }]
    """
    print('牙齿分割完毕！')
    # 识别牙齿编号
    no_num_idx = []  # 还没有编号的牙齿索引会存放在这里
    num_list = []  # 已经编号的牙齿
    for idx, tooth in enumerate(tooth_list):
        num = classify_tooth(tooth)
        num_filter = list(
            filter(lambda x: x == num, num_list))
        if len(num_filter) != 0:  # 如果已经有了牙齿编号
            no_num_idx.append(idx)
            continue
        slice_data[idx]['num'] = float(num)
        num_list.append(num)
    # 给识别重复的牙齿随机一个编号，编号不能重复，可以错误，否则前端无法正确手动编号！

    # 还没有分配的牙齿编号(目前可分配的牙齿编号)
    tem_list = [x for x in teeth_list if x not in num_list]
    if len(no_num_idx) != 0:
        for i, num_idx in enumerate(no_num_idx):
            # 这里要区分上下颌,目前还没有区分
            slice_data[num_idx]['num'] = float(tem_list[i])
            num_list.append(tem_list[i])

    print('开始组装！')
    res_sitk, out_result = combination(teeth_data, slice_data)

    # 判断是否缺牙
    out_result[out_result != 0] = 1
    redundant = np.sum(out_result)/np.sum(two_out)
    if redundant < 0.985:  # 可能存在缺牙
        mask = out_result != 0
        two_out[mask] = 0
        sitk_out = sitk.GetImageFromArray(two_out.astype(np.uint8))
        out_open = sitk.BinaryMorphologicalOpening(
            sitk_out != 0)  # 经过一次开运算
        redundant_out = sitk.GetArrayFromImage(out_open)
        sitk.WriteImage(out_open, os.path.join(
            r'D:\jiang_paper\fantest\other_two', fname))
        # 再次寻找牙齿
        res, _ = div_teeth(redundant_out, [], num_list)
        for value in res:
            slice_data.append(value)
        res_sitk, out_result = combination(teeth_data, slice_data)

    # 将原始图像信息还原
    res_sitk.SetSpacing(space)
    res_sitk.SetOrigin(origin)
    res_sitk.SetDirection(direction)
    print('组装完毕！')
    # 保存

    sitk.WriteImage(res_sitk, os.path.join(save_path, fname))


def div_teeth(two_out, no_num_idx, num_list):
    """

    :params no_num_idx: 还没有编号的牙齿索引会存放在这里
    :params num_list: 已经编号的牙齿
    :return:slice_data and num_list
    """

    res_teeth_box = detect(two_out)  # 牙齿候选区数据
    slice_data = generate_slice_teeth(res_teeth_box, two_out)  # 提取单颗牙
    """
       slice_data = [{
        tooth:..., 切片下来的单颗牙齿
        boundary:..., 牙齿边框
        num:..., 编号
        ori_shape_tooth:..., 每一颗牙齿的原始形状
        center:..., 中心点
        }]
       """
    print('开始分割！')
    tooth_list = start_seg(slice_data)
    print('牙齿分割完毕！')
    # 识别牙齿编号
    for idx, tooth in enumerate(tooth_list):
        num = classify_tooth(tooth)
        num_filter = list(
            filter(lambda x: x == num, num_list))
        if len(num_filter) != 0:  # 如果已经有了牙齿编号
            no_num_idx.append(idx)
            continue
        slice_data[idx]['num'] = float(num)
        num_list.append(num)
    # 给识别重复的牙齿随机一个编号，编号不能重复，可以错误，否则前端无法正确手动编号！
    # 还没有分配牙齿的编号(目前可分配的牙齿编号)
    tem_list = [x for x in teeth_list if x not in num_list]
    if len(no_num_idx) != 0:
        for i, num_idx in enumerate(no_num_idx):
            # 这里要区分上下颌,目前还没有区分
            slice_data[num_idx]['num'] = float(tem_list[i])
            num_list.append(tem_list[i])
    return slice_data, num_list
