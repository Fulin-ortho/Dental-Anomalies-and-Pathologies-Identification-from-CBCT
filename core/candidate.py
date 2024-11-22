import torch
from net.net3d import UNet
import SimpleITK as sitk
import numpy as np
import math
from utils.cbct_utils import resize_image_itk
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('正在加载候选区网络...')
center_net = UNet(1, 4, 16).eval().to(device)
center_net.load_state_dict(torch.load(
    r'./your_weights', map_location=device, weights_only=True))
print('候选区网络加载完毕...')
pool = torch.nn.MaxPool3d(kernel_size=3, stride=3, padding=1)


def detect(ori_np, open=True, k=60, threshold=0.08):
    """
    检测牙齿候选区
    :params ori_np:二分类的整口牙齿数据 
    :params k: 选取前最大的多少个初始值，默认为60
    :params threshold: 中心点置信度，默认为0.1
    :return:牙齿的包围盒数组[[x,y,z,w,h,d],...]
    """
    ori_data = sitk.GetImageFromArray(ori_np.astype(np.uint8))
    if open:
        ori_data = sitk.BinaryMorphologicalOpening(ori_data != 0, 3)
    size = ori_data.GetSize()  # x,y,z
    x_scale, y_scale, z_scale = size[0]/128, size[1]/128, size[2]/128
    data_np = resize_image_itk(ori_data, (128, 128, 128), True)
    data_np[data_np != 0] = 1  # 二值化数据
    data_np = data_np.reshape(1, 1, 128, 128, 128)

    input_data = torch.from_numpy(data_np).type(torch.FloatTensor).to(device)
    out = center_net(input_data)
    heatmap = torch.sigmoid(out[:, 0, ...])

    whd = out[:, 1:, ...].squeeze().cpu().data.numpy()
    boxes = screen(heatmap, whd, k, threshold)
    bias = 10  # 将候选区的维度扩大，以便于包裹住整颗牙齿
    # 还原
    for box in boxes:
        box[0] = int(box[0] * x_scale)
        box[1] = int(box[1] * y_scale)
        box[2] = int(box[2] * z_scale)
        box[3] = int(box[3] * x_scale+bias)
        box[4] = int(box[4] * y_scale+bias)
        box[5] = int(box[5] * z_scale+bias)
    return boxes


def screen(heatmap, whd, k, threshold):
    """
    对heatmap中的值进行筛选操作，选出达到条件的中心点
    :params heatmap: 中心点的热图
    :params whd: 网络预测的长宽高
    :params k: 选取前最大的多少个初始值
    :params threshold: 中心点置信度
    :return:候选区的box [x,y,z,w,h,d]
    """

    down = pool(heatmap).squeeze()  # 经过最大池化操作，选取每个点最大的值

    down_viwe = down.view(-1, down.shape[0]*down.shape[1]*down.shape[2])

    top_k = down_viwe.topk(k)
    top_k = top_k[0].squeeze().cpu().data.numpy()

    heatmap = heatmap.squeeze().cpu().data.numpy()
    boxes = []
    for value in top_k:
        if value <= threshold:
            continue
        a = np.argwhere(heatmap == value).squeeze()

        d = whd[0, a[0], a[1], a[2]]
        h = whd[1, a[0], a[1], a[2]]
        w = whd[2, a[0], a[1], a[2]]
        box = [a[2], a[1], a[0], w, h, d]  # x,y,z,w,h,d
        # 去重
        if len(boxes) > 0:
            d = min_distance(boxes, [a[2], a[1], a[0]])

            if d <= 5:
                continue
        boxes.append(box)

    return boxes


def min_distance(array, num):
    """
    比较一个数据与数组中的数据的比较的最小值，并且返回最小值
    :params array:需要比较的数组 
    :params num:需要和数组比较的值 
    :return:最小值
    """
    res = list(map(lambda a: math.sqrt(
        pow((a[0]-num[0]), 2)+pow((a[1]-num[1]), 2)+pow((a[2]-num[2]), 2)), array))

    return abs(min(res))


def generate_slice_teeth(boxes, array):
    """
    将整副牙齿进行单颗牙齿切片
    :params boxes: 单颗牙齿数据(x,y,z,w,h,d)
    :params array: 二分类的结果或者原图三维图像
    :return:切片下来的牙齿数据的数组
    """
    data = []
    for box in boxes:
        x, y, z, w, h, d = box[0], box[1], box[2], box[3], box[4], box[5]
        zmin, zmax = int(z-d/2) if int(z-d/2) > 0 else 0, int(z+d/2)
        ymin, ymax = int(y-h/2) if int(y-h/2) > 0 else 0, int(y+h/2)
        xmin, xmax = int(x-w/2) if int(x-w/2) > 0 else 0, int(x+w/2)
        ana = array[zmin:zmax, ymin:ymax, xmin:xmax]
        data.append({
            'tooth': ana,
            'boundary': [zmin, zmax, ymin, ymax, xmin, xmax],
            'center': (x, y, z),
            'num': None
        })
    return data
