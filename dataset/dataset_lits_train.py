from torch.utils.data import DataLoader
import os
import sys
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from .data_augmentation import random_augmentation
import numpy as np





class Train_Dataset(dataset):
    def __init__(self, args):

        self.args = args

        self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))




    def __getitem__(self, index):

        ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkFloat32)
        ct_array = sitk.GetArrayFromImage(ct)
        seg_1 = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)
        seg_2 = sitk.ReadImage(self.filename_list[index][2], sitk.sitkUInt8)
        seg_1 = sitk.GetArrayFromImage(seg_1)
        seg_2 = sitk.GetArrayFromImage(seg_2)

        ct_array, seg_array1, seg_array2 = random_augmentation(ct_array, seg_1, seg_2, (16, 16, 16), (0.75, 1.25), (0.7, 1.5), 0.5)
        ct_array = ct_array.copy()
        seg_array1 = seg_array1.copy()
        seg_array2 = seg_array2.copy()
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array1 = torch.FloatTensor(seg_array1).unsqueeze(0)
        seg_array2 = torch.FloatTensor(seg_array2).unsqueeze(0)



        loss_weight = []
        # if self.filename_list[index][0].find("liver") != -1:
        #     loss_weight = [0.8,0.1,0.1]
        # elif self.filename_list[index][0].find("kindey") != -1:
        #     loss_weight = [0.1,0.8,0.1]
        # else:
        #     loss_weight = [0.1, 0.1, 0.8]
        return ct_array, [seg_array1.squeeze(), seg_array2.squeeze()]

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list

if __name__ == "__main__":
    sys.path.append('/ssd/lzq/3DUNet')
    from config import args
    train_ds = Train_Dataset(args)

    # 定义数据加载
    train_dl = DataLoader(train_ds, 73, False, num_workers=1)

    # for i, (ct, seg) in enumerate(train_dl):
    #     print(i,ct.size(),seg.size())