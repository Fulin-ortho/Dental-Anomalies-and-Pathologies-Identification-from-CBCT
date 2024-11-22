import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


class BoxTeethDataset(Dataset):
    def __init__(self, img_list, label_list, transform=None):
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx], 0)

        if self.transform:
            img = self.transform(img)
        label = np.load(self.label_list[idx])
        return img, torch.from_numpy(label)


class DefectTeethDataset(Dataset):
    def __init__(self, img_list, label_list, transform=None):
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx], 0)
        if self.transform:
            img = self.transform(img)
        label = np.load(self.label_list[idx])
        label = label.squeeze()
        return img, torch.from_numpy(label)
