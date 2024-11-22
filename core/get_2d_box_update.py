from torchvision.transforms import transforms
import torch
from net.net2d import MyResBoxNet
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = os.path.dirname(os.path.dirname(__file__))
print('正在初始化2d box网络......')
teeth_box_net_z = MyResBoxNet(4).to(device)  # 加载横断面网络
teeth_box_net_z.eval()
teeth_box_net_z.load_state_dict(torch.load(
    r'./your_weights', map_location=device, weights_only=True))

teeth_box_net_y = MyResBoxNet(4).to(device)  # 加载冠状面网络
teeth_box_net_y.eval()
teeth_box_net_y.load_state_dict(torch.load(
    r'/your_weights', map_location=device, weights_only=True))
print('2d box初始化完成!')


def get_teeth_box(z_img, y_img, shape, bias=10):
    ori_z, ori_y, ori_x = shape
    out_z = detect_2d_box(z_img)
    out_y = detect_2d_box(y_img, type='y')
    x_min = out_z[0] - bias if out_z[0] - bias >= 0 else 0
    x_max = out_z[2] + bias if out_z[2] + bias <= ori_x else ori_x
    y_min = out_z[1] - bias if out_z[1] - bias >= 0 else 0
    y_max = out_z[3] + bias if out_z[3] + bias <= ori_y else ori_y
    z_min = out_y[1] - bias if out_y[1] - bias >= 0 else 0
    z_max = out_y[3] + bias if out_y[3] + bias <= ori_z else ori_z
    return [z_min, z_max, y_min, y_max, x_min, x_max]


def detect_2d_box(img, type='z'):
    y, x = img.shape
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224])
    ])

    input_img = transform(img).unsqueeze(0).type(torch.FloatTensor)
    if type == 'z':
        out = teeth_box_net_z(input_img.to(
            device)).cpu().data.numpy().squeeze(0)
        x_min, y_min, x_max, y_max = int(
            out[0] * x), int(out[1] * y), int(out[2] * x), int(out[3] * y)
        return [x_min, y_min, x_max, y_max]
    if type == 'y':
        out = teeth_box_net_y(input_img.to(
            device)).cpu().data.numpy().squeeze(0)
        x_min, y_max, x_max, y_min = int(
            out[0] * x), int(out[1] * y), int(out[2] * x), int(out[3] * y)
        return [x_min, y_min, x_max, y_max]
