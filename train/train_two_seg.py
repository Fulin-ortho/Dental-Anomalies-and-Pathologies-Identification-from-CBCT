import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *
from net.net3d import UNet
from data_loader.data_loader3d import TwoSegSliding
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练分割单颗牙齿网络1 利用滑窗的形式训练同一个数据
# 整口牙齿的二分割训练和这个比较类似，只是输入数据的形状不同，以及网络参数的区别，不再额外展示


def train_nn1():
    lr = 0.001
    ite_num = 4000
    save_frq = 2000
    batch_size = 6
    save_path = r'./your_path/'
    # 第一次分割的网络,输出置信度以及初步的分割结果
    model_1 = UNet(1, 2, 16).to(device)
    model_1.train()

    criterion = nn.BCEWithLogitsLoss(reduction='none')  # 切割损失
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=lr)

    # nii路径
    nii_list = glob.glob(r'your_path\*\*')
    # label 路径
    label_list = glob.glob(r'your_path\*\*')

    dataset = TwoSegSliding(
        img_list=nii_list, label_list=label_list, size=(96, 64, 64))
    loader = DataLoader(dataset=dataset, shuffle=True,
                        batch_size=batch_size, num_workers=4)
    for epoch in range(100000):
        for _, (data, target) in enumerate(loader):
            ite_num += 1

            data = data.type(torch.FloatTensor).to(device)

            target = target.type(torch.FloatTensor).to(device)
            out = model_1(data)

            out_0 = out[:, 0, :, :, :]  # 切割结果 mask
            out_0 = out_0.unsqueeze(dim=1)

            out_1 = torch.sigmoid(out[:, 1, :, :, :])  # 置信度
            out_unsqueeze = out_1.unsqueeze(dim=1)

            rough_loss = criterion(out_0, target)  # 初步切割的损失
            loss_s1 = confidence(rough_loss, out_unsqueeze)  # 第一步的损失

            # 更新网络1的参数
            optimizer_1.zero_grad()
            loss_s1.backward()
            optimizer_1.step()

            if ite_num % 20 == 0:
                print('epoch:', epoch, 'loss_s1:', loss_s1.item(), 'ite', ite_num, 'rough_loss:',
                      rough_loss.mean().item())
            if ite_num % save_frq == 0:
                torch.save(model_1.state_dict(),
                           save_path + 'model1_ite_num_%d_loss_%3f.pth' % (ite_num, loss_s1.item()))

    torch.save(model_1.state_dict(), save_path + 'model1_result.pth')


# 训练分割单颗牙齿精细分割网络2
def train_nn2():
    lr = 0.001
    ite_num = 0
    save_frq = 2000
    batch_size = 6
    save_path = r'./your_path/'

    model_1 = UNet(1, 2, 16).to(device)
    model_1.eval()

    model_2 = UNet(2, 1, 16).to(device)
    model_2.train()

    criterion_2 = nn.BCEWithLogitsLoss(reduction='none')
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=lr)

    # nii路径
    nii_list = glob.glob(r'your_path\*\*')
    # label 路径
    label_list = glob.glob(r'your_path\*\*')

    dataset = TwoSegSliding(
        img_list=nii_list, label_list=label_list, size=(96, 64, 64))
    loader = DataLoader(dataset=dataset, shuffle=True,
                        batch_size=batch_size, num_workers=4)

    for epoch in range(100000):
        for _, (data, target) in enumerate(loader):
            ite_num += 1
            data = data.type(torch.FloatTensor).to(device)  # [-1,1]
            target = target.type(torch.FloatTensor).to(device)
            with torch.no_grad():
                out = torch.sigmoid(model_1(data))
                out_mask = out[:, 0, :, :, :]  # 切割结果 mask [0,1]
                out_mask = out_mask.unsqueeze(dim=1)

                out_conf = out[:, 1, :, :, :]  # 置信度 [0,1]
                out_conf = out_conf.unsqueeze(dim=1)
            # 拼接处理
            input_2 = torch.cat((data, out_mask), dim=1)

            s2_out = model_2(input_2)
            s2_loss = criterion_2(s2_out, target)
            w = 1 - out_conf

            loss_s2 = torch.mean((1 + w) * s2_loss)

            # 更新网络2的参数
            optimizer_2.zero_grad()
            loss_s2.backward()
            optimizer_2.step()

            if ite_num % 20 == 0:
                print('epoch:', epoch, 'loss_s2:', loss_s2.item(),
                      'bce_loss: ', s2_loss.mean().item(), 'ite', ite_num)
            if ite_num % save_frq == 0:
                torch.save(model_2.state_dict(),
                           save_path + 'model2_ite_num_%d_loss_%3f.pth' % (ite_num, loss_s2.item()))

    torch.save(model_2.state_dict(), save_path + 'model2_result.pth')


def confidence(bce_loss, conf_out):
    Loss_S1 = torch.mean(torch.pow(bce_loss * conf_out,
                         2) + torch.pow((1 - conf_out), 2))
    return Loss_S1
