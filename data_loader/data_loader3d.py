import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import torch
from utils.cbct_utils import *
from utils.data_utils import *


class ObjDataset(Dataset):
    def __init__(self, img_list, label_list):
        self.img_list = img_list
        self.label_list = label_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, label = self.get_train_batch_by_index(idx)
        return torch.from_numpy(img), torch.from_numpy(label)

    def get_train_batch_by_index(self, index):
        img_np = sitk_read_raw(self.img_list[index])
        # 将数据置为[-1,1]之间
        ab = np.abs(img_np)  # 求绝对值
        img_np = img_np / np.max(ab)

        label_np = sitk_read_raw(self.label_list[index])

        return np.expand_dims(img_np, axis=0), np.expand_dims(label_np, axis=0)


class Classify_cutting(Dataset):
    def __init__(self, img_list, label_list):
        self.img_list = img_list
        self.label_list = label_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, label = self.get_train_batch_by_index(idx)
        return torch.from_numpy(img), torch.from_numpy(label)

    def get_train_batch_by_index(self, index):
        img_np = sitk_read_raw(self.img_list[index])
        # 将数据置为[-1,1]之间
        ab = np.abs(img_np)  # 求绝对值
        img_np = img_np / np.max(ab)

        label_np = np.load(self.label_list[index])
        return np.expand_dims(img_np, axis=0), label_np


class Box_Classify(Dataset):
    def __init__(self, img_list, label_list):
        self.img_list = img_list
        self.label_list = label_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, label = self.get_train_batch_by_index(index)
        return torch.from_numpy(img), torch.from_numpy(label)

    def get_train_batch_by_index(self, index):
        img_np = sitk_read_raw(self.img_list[index])
        # 将数据置为[-1,1]之间
        ab = np.abs(img_np)  # 求绝对值
        img_np = img_np / np.max(ab)

        label_np = np.load(self.label_list[index])
        return np.expand_dims(img_np, axis=0), label_np


class Class_IsTooth(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = sitk_read_raw(self.data_list[index])
        label = self.data_list[index].split('\\')
        label = int(label[-2])
        label = torch.LongTensor([label])
        return np.expand_dims(img, axis=0), label


# 整颗牙齿切割，用my resize
class Whole_Single_cutting(Dataset):
    def __init__(self, img_list, label_list, size):
        self.img_list = img_list
        self.label_list = label_list
        self.size = size  # (x,y,z)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, label = self.get_train_batch_by_index(idx)
        return torch.from_numpy(img), torch.from_numpy(label)

    def get_train_batch_by_index(self, index):
        itk_img = sitk.ReadImage(self.img_list[index])
        img_np = resize_image_itk(itk_img, self.size, resamplemethod=sitk.sitkLinear)
        # 将数据置为[-1,1]之间
        ab = np.abs(img_np)  # 求绝对值
        img_np = img_np / np.max(ab)

        itk_label = sitk.ReadImage(self.label_list[index])
        label_np = resize_image_itk(itk_label, self.size, resamplemethod=sitk.sitkNearestNeighbor)

        return np.expand_dims(img_np, axis=0), np.expand_dims(label_np, axis=0)


# 使用滑动窗口的形式，随机从CBCT里面取一块进行训练
class TwoSegSliding(Dataset):
    def __init__(self, img_list, label_list, size):
        '''
        :param img_list:原始数据
        :param label_list: 标签数据
        :param size: 需要截取的形状（z,y,x）
        '''
        self.img_list = img_list
        self.label_list = label_list
        self.size = size

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, label = self.get_train_batch_by_index(idx)
        return torch.from_numpy(img), torch.from_numpy(label)

    def get_train_batch_by_index(self, index):
        img_np = sitk_read_raw(self.img_list[index])
        # 将数据置为[-1,1]之间
        ab = np.abs(img_np) 
        if np.max(ab) > 0:
            img_np = img_np / np.max(ab)
        label_np = sitk_read_raw(self.label_list[index])

        img, label = random_crop_3d(img=img_np, label=label_np, crop_size=self.size)

        return np.expand_dims(img, axis=0), np.expand_dims(label, axis=0)


class Single_cutting(Dataset):
    def __init__(self, img_list, label_list):
        self.img_list = img_list
        self.label_list = label_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, label = self.get_train_batch_by_index(idx)
        return torch.from_numpy(img), torch.from_numpy(label)

    def get_train_batch_by_index(self, index):
        try:
            img_np = sitk_read_raw(self.img_list[index])
            
            # 将数据置为[-1,1]之间
            ab = np.abs(img_np)  
            img_np = img_np / np.max(ab)

            label_np = sitk_read_raw(self.label_list[index])


            img, label = resize(img_np, (96, 96)), resize(label_np, (96, 96))
            img, label = resize_zy(img, (96, 64)), resize_zy(label, (96, 64))
        except:
            print(self.img_list[index], self.label_list[index])
            exit()
        return np.expand_dims(img, axis=0), np.expand_dims(label, axis=0)


