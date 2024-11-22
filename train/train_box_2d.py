import torch
import torch.nn as nn
from net.net2d import BoxNet2D
import glob
from torch.utils.data import DataLoader
from data_loader.data_loader2d import BoxTeethDataset
from torchvision.transforms import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2D网络找包围盒


def train():
    lr = 0.001
    ite_num = 1000
    save_frq = 500
    batch_size = 128
    save_path = r'./your_path/'
    net = BoxNet2D(1, 4, 16).to(device)
    net.train()

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # 需要从两个纬度寻找包围盒，这里以z轴为例
    data_list = glob.glob(r'your_path\data_z\*')
    label_list = glob.glob(r'your_path\label_z\*')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(),
        transforms.Resize([128, 128])
    ])

    dataset = BoxTeethDataset(
        img_list=data_list, label_list=label_list, transform=transform)
    loader = DataLoader(dataset=dataset, shuffle=True,
                        batch_size=batch_size, num_workers=0)

    for epoch in range(10000):
        for _, (data, target) in enumerate(loader):
            ite_num += 1
            data = data.type(torch.FloatTensor).to(device)
            target = target.type(torch.FloatTensor).to(device)
            out = net(data)
            loss = criterion(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ite_num % 20 == 0:
                print('epoch:', epoch, 'loss:', loss.item(), 'ite', ite_num)
            if ite_num % save_frq == 0:
                torch.save(net.state_dict(),
                           save_path + 'net_z_ite_num_%d_loss_%3f.pth' % (ite_num, loss.item()))
    torch.save(net.state_dict(), save_path + 'net_z_result.pth')
