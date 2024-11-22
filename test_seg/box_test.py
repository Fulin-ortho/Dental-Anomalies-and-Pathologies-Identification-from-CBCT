from torchvision.transforms import transforms
from net.net2d import BoxNet2D
import torch
import cv2
import glob
from core.get_2d_box_update import detect_2d_box as box

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('正在初始化2d box网络......')
teeth_box_net_z = BoxNet2D(1, 4, 16).to(device)  # 加载横断面网络
teeth_box_net_z.eval()
teeth_box_net_z.load_state_dict(torch.load(
    r'your_weights'))

teeth_box_net_y = BoxNet2D(1, 4, 16).to(device)  # 加载冠状面网络
teeth_box_net_y.eval()
teeth_box_net_y.load_state_dict(torch.load(
    r'your_weights'))
print('2d box初始化完成！')


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
    elif type == 'y':
        out = teeth_box_net_y(input_img.to(
            device)).cpu().data.numpy().squeeze(0)

    x_min, y_min, x_max, y_max = int(
        out[0]*x), int(out[1]*y), int(out[2]*x), int(out[3]*y)
    my_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(my_img, (x_min - 10, y_min - 10),
                  (x_max + 10, y_max + 10), (0, 0, 255), 2)  # 红

    cv2.namedWindow("img_ori")
    cv2.imshow('img_ori', my_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_path = glob.glob(r'D:\jiang_paper\fantest\img_y\*')
    for p in img_path:
        img = cv2.imread(
            p, 0)

        res = box(img, type='y')
        x_min, y_min, x_max, y_max = res[0], res[1], res[2], res[3]
        my_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(my_img, (x_min - 10, y_min - 10),
                      (x_max + 10, y_max + 10), (0, 0, 255), 2)  # 红

        cv2.namedWindow("img_ori")
        cv2.imshow('img_ori', my_img)
        cv2.waitKey(0)
