
import torch
from net.resnet3d import generate_model
from torch.utils.data import DataLoader
import glob
from data_loader.data_loader_classify import MyDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 牙齿分类


def tarin():
    lr_start = 0.001
    save_frq = 20
    batch_size = 2
    save_path = ''
    ite_num = 0
    accumulation_steps = 1
    net = generate_model(50, n_input_channels=1, n_classes=32).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_start)
    loss_func = torch.nn.CrossEntropyLoss()
    # train dataset
    data_list = glob.glob(r'your_path\train\*\*')
    dataset = MyDataset(data_list)
    loader = DataLoader(dataset=dataset, shuffle=True,
                        batch_size=batch_size)

    # val dataset
    val_data_list = glob.glob(r'your_path\val\*\*')
    val_dataset = MyDataset(val_data_list)
    val_loader = DataLoader(dataset=val_dataset, shuffle=True,
                            batch_size=batch_size)

    for epoch in range(100):
        net.train()
        total_train_loss = []
        for i, (data, target) in enumerate(loader):
            ite_num = ite_num+1
            data = data.type(torch.FloatTensor).to(device)
            target = target.squeeze(1).to(device)
            out = net(data)

            loss = loss_func(out, target)
            loss = loss/accumulation_steps
            total_train_loss.append(float(loss.item()))
            loss.backward()
            if (i + 1) % accumulation_steps == 0:  # 使用梯度累加，变相增大batch_size
                optimizer.step()
                optimizer.zero_grad()

            if ite_num % 20 == 0:
                print('epoch:', epoch, 'item:', ite_num, 'loss:', loss.item())
            if ite_num % save_frq == 0:
                torch.save(net.state_dict(), save_path +
                           'epoch_%d_loss_%3f.pth' % (epoch, loss.item()))
        if epoch % 1 == 0:
            with torch.no_grad():
                val_loss_list = []
                accuracy_list = []
                net.eval()
                for _, (val_data, val_label) in enumerate(val_loader):
                    val_data = val_data.type(torch.FloatTensor).to(device)
                    val_label = val_label.squeeze(1).to(device)
                    val_out = net(val_data)  # 预测值

                    val_loss = loss_func(val_out, val_label)
                    val_loss_list.append(float(val_loss.item()))
                    # 算准确率
                    pred = torch.max(val_out, 1)[1].cpu().data.numpy()
                    accuracy = (pred == val_label.cpu().data.numpy()
                                ).astype(int).sum() / batch_size
                    accuracy_list.append(accuracy)

                    print('epoch:', epoch, 'val_loss:', val_loss.item())
                val_loss_mean = sum(val_loss_list) / len(val_loss_list)
                train_loss_mean = sum(
                    total_train_loss) / len(total_train_loss)
                accuracy_mean = sum(accuracy_list)/len(accuracy_list) * 100
                print('epoch:', epoch, 'train_loss:',
                      train_loss_mean, 'val_loss:', val_loss_mean, 'acc:', accuracy_mean)
                # 写入本地文件
                with open(r'your_path/my_resnet_loss_acc.txt', 'a') as f:
                    f.write(str(epoch) + ' ' + str(train_loss_mean) +
                            ' ' + str(val_loss_mean) + ' ' + str(accuracy_mean) + '\n')
        torch.save(net.state_dict(), save_path + 'final.pth')
