import os
import numpy as np
from torch.utils.data import Dataset
import torch
from utils.cbct_utils import sitk_read_raw


class MyDataset(Dataset):
    """
    目标检测网络的自定义dataset
    """

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_np = sitk_read_raw(self.data_list[index])  # [0,1]
        data_np = np.expand_dims(data_np, axis=0)
        # 标签
        path, _ = os.path.split(self.data_list[index])
        _, n = os.path.split(path)
        label = np.array([int(n)])
        return torch.from_numpy(data_np), torch.tensor(label).type(torch.LongTensor)
