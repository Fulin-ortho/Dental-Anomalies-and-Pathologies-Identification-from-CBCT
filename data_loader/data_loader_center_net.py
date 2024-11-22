from torch.utils.data import Dataset
import torch
import numpy as np
from utils.cbct_utils import sitk_read_raw


class DetectDataset(Dataset):
    """
    目标检测网络的自定义dataset
    """

    def __init__(self, data_list, label_list):
        self.data_list = data_list
        self.label_list = label_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_np = sitk_read_raw(self.data_list[index])  # [0,1]
        label_np = np.load(self.label_list[index])
        label_np = label_np['data']

        return torch.from_numpy(data_np), torch.from_numpy(label_np)
