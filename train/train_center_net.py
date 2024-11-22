import torch
from net.net3d import UNet
from torch.utils.data import DataLoader
import glob
from data_loader.data_loader_center_net import DetectDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_loss = torch.nn.SmoothL1Loss()

# 牙齿质心


def train():
    lr_start = 0.001
    save_frq = 20
    batch_size = 6
    save_path = r'your_path'
    net = UNet(1, 13, 16).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_start)
    # train dataset
    data_list = glob.glob('')
    label_list = glob.glob('')
    dataset = DetectDataset(data_list, label_list)
    loader = DataLoader(dataset=dataset, shuffle=True,
                        batch_size=batch_size, num_workers=4)

    # val dataset
    val_data_list = glob.glob()
    val_label_list = glob.glob('')
    val_dataset = DetectDataset(val_data_list, val_label_list)
    val_loader = DataLoader(dataset=val_dataset,
                            shuffle=True, batch_size=batch_size, num_workers=4)
    # start trian
    for epoch in range(10000):
        net.train()
        total_train_loss = []
        for _, (data, label) in enumerate(loader):
            data = data.type(torch.FloatTensor).to(device)
            label = label.type(torch.FloatTensor).to(device)
            out = net(data)
            loss = loss_func(label, out)
            total_train_loss.append(float(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0:
                print('epoch:', epoch, 'loss:', loss.item())
            if epoch % save_frq == 0:
                torch.save(net.state_dict(), save_path +
                           'epoch_%d_loss_%3f.pth' % (epoch, loss.item()))
        if epoch % 2 == 0:
            # 验证
            with torch.no_grad():
                val_loss_list = []
                net.eval()
                for _, (val_data, val_label) in enumerate(val_loader):
                    val_data = val_data.type(torch.FloatTensor).to(device)
                    val_label = val_label.type(torch.FloatTensor).to(device)
                    val_out = net(val_data)
                    val_loss = loss_func(val_label, val_out)
                    val_loss_list.append(float(val_loss.item()))
                val_loss_mean = sum(val_loss_list)/len(val_loss_list)
                train_loss_mean = sum(total_train_loss)/len(total_train_loss)
                print('epoch:', epoch, 'train_loss:',
                      train_loss_mean, 'val_loss:', val_loss_mean)
                # 写入本地文件
                with open(r'', 'a') as f:
                    f.write(str(epoch)+' ' + str(train_loss_mean) +
                            ' '+str(val_loss_mean))
    torch.save(net.state_dict(), save_path+'final.pth')


def loss_func(target, prediction, is_off=True):
    """
    计算网络损失
    :params target: 目标值
    :params prediction: 预测值
    :params is_off: 是否计算中心偏移损失，默认为false
    :return:中心损失+中心偏移损失+尺度损失
    """
    # 定义中心偏移权重损失权重和尺寸损失权重
    l_off, l_size = 1, 0.1
    # size loss 只计算整样本的损失
    positive_mask = torch.gt(target[:, 4:7, ...], 0)
    target_size = target[:, 1:4, ...][positive_mask]
    prediction_size = prediction[:, 1:4, ...][positive_mask]
    loss_size = l_size * base_loss(target_size, prediction_size)
    if is_off:
        # center_off loss 只计算正样本的损失
        positive_mask = torch.gt(target[:, 10:13, ...], 0)
        target_off = target[:, 7:10, ...][positive_mask]
        prediction_off = prediction[:, 7:10, ...][positive_mask]
        loss_off = l_off * base_loss(target_off, prediction_off)

    # center loss 使用改进的focal loss
    alpha = 2
    beta = 4
    heatmap, heatmap_p = target[:, 0, ...], torch.sigmoid(
        prediction[:, 0, ...])  # 预测的热图要在[0-1]之间
    # 计算正负样本mask
    positive_mask = torch.eq(heatmap, 1)  # 正样本,label 为1
    negative_mask = torch.lt(heatmap, 1)  # 其他情况，label不为1

    center_t = heatmap
    center_p = heatmap_p

    loss_positive_center = torch.pow(
        (1 - center_p), alpha) * torch.log(center_p)  # 正样本损失
    loss_positive_center = loss_positive_center[positive_mask].mean()

    loss_negative_center = torch.pow(
        (1 - center_t), beta) * torch.pow(center_p, alpha) * torch.log(1 - center_p)  # 其他情况
    loss_negative_center = loss_negative_center[negative_mask].mean()
    heatmap_loss = -(loss_negative_center + loss_positive_center)
    if is_off:
        return heatmap_loss + loss_size + loss_off
    return heatmap_loss + loss_size
