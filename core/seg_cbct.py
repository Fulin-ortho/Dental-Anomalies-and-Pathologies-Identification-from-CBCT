import SimpleITK as sitk
import torch
from net.net3d import UNet
from net.resnet3d import generate_model
from utils.cbct_utils import resize_image_itk
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('正在加载seg模块网络！')
seg_net = UNet(1, 2, 16).to(device).eval()
seg_net.load_state_dict(torch.load(
    r'./your_weights', map_location=device, weights_only=True))

seg_net_2 = UNet(2, 1, 16).to(device).eval()
seg_net_2.load_state_dict(torch.load(
    r'./your_weights', map_location=device, weights_only=True))

class_net = generate_model(50, n_input_channels=1,
                           n_classes=49).eval().to(device)
class_net.load_state_dict(torch.load(
    r'./your_weights', map_location=device, weights_only=True))
print('seg模块网络加载完毕！')


def start_seg(ana_data):
    """
    :params ana_data:牙齿数据 
    :return:分割之后的单颗牙齿数组
    """
    tooth_list = []  # 分割之后的牙齿,形状没有恢复
    for ana in ana_data:
        slice = ana['tooth']
        itk_img = sitk.GetImageFromArray(slice.astype('uint8'))
        input_img = resize_image_itk(itk_img, (64, 64, 96), True)
        tooth, ori_shape_tooth = seg_single(input_img, slice.shape)
        ana['tooth'] = tooth
        ana['ori_shape_tooth'] = ori_shape_tooth
        tooth_list.append(tooth)
    return tooth_list


def seg_single(img, shape):
    """
    分割单颗牙齿
    :params img:网络输入图像 
    :params shape:网络输入图像的原始形状
    :return:网络的输出结果；输出结果形状还原的数据
    """
    z, y, x = shape
    img = img.reshape(1, 1, 96, 64, 64)
    input_img = torch.from_numpy(img).type(torch.FloatTensor).to(device)
    with torch.no_grad():
        out = torch.sigmoid(seg_net(input_img))
    mask = out[:, 0, :, :, :]
    mask = mask.unsqueeze(dim=1)

    input_img_2 = torch.cat((input_img, mask), 1)
    with torch.no_grad():
        net2_out = torch.sigmoid(seg_net_2(input_img_2))

    net2_out = net2_out.squeeze().cpu().data.numpy()
    net2_out[net2_out <= 0.5] = 0
    net2_out[net2_out > 0.5] = 1

    itkimgResampled = sitk.GetImageFromArray(net2_out.astype('uint8'))
    ori = resize_image_itk(itkimgResampled, (x, y, z), True)
    return net2_out, ori  # (96,64,64) ,原始形状


def classify_tooth(tooth_np):
    """
    对牙齿进行分类
    :params tooth_np: 单颗牙齿的numpy数据，形状(96,64,64)
    :return:牙齿编号
    """
    data_np = np.expand_dims(tooth_np, axis=0)
    data_np = data_np.reshape(1, 1, 96, 64, 64)
    input_tooth = torch.from_numpy(data_np).type(
        torch.FloatTensor).to(device)
    pred = torch.softmax(class_net(input_tooth), 1)
    pred = torch.max(pred, 1)[1].cpu().data.numpy()  # 预测的编号

    return pred[0]


def combination(teeth_data, slice_data):
    ori_shape = teeth_data['ori_shape']  # 原始数据形状
    cutting_shape = teeth_data['cutting_data'].shape  # 切片数据形状
    out_result = np.zeros(cutting_shape)
    for obj in slice_data:
        out_np = obj['ori_shape_tooth']
        boundary = obj['boundary']
        nii = np.zeros(cutting_shape)
        label_zero = np.zeros(out_np.shape)
        label_zero[out_np == 1] = obj['num']

        nii[boundary[0]:boundary[1], boundary[2]:boundary[3],
            boundary[4]:boundary[5]] = label_zero

        out_result = out_result + nii
        # 去掉重复区域
        mask = np.logical_and(out_result, nii)
        out_result = out_result + nii

        out_result[mask] = obj['num']
    # 恢复到输入cbct的形状
    final_res = np.zeros(ori_shape)
    teeth_box = teeth_data['teethBox']
    final_res[teeth_box[0]:teeth_box[1], teeth_box[2]:teeth_box[3], teeth_box[4]:teeth_box[5]] = out_result
    final_res_itk = sitk.GetImageFromArray(final_res.astype('uint8'))
    return final_res_itk, out_result
