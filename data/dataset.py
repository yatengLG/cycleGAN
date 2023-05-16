# -*- coding: utf-8 -*-
# @Author  : LG

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os


class Dataset_CycleGAN(Dataset):
    def __init__(self, A_root, B_root, transform_A=None, transform_B=None):
        self.transform_A = transform_A
        self.transform_B = transform_B
        self.A_paths = [os.path.join(A_root, f) for f in os.listdir(A_root)]
        self.B_paths = [os.path.join(B_root, f) for f in os.listdir(B_root)]
        self.A_len = len(self.A_paths)
        self.B_len = len(self.B_paths)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_len]
        B_path = self.B_paths[index % self.B_len]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        if self.transform_A is not None:
            A_img = self.transform_A(A_img)
        if self.transform_B is not None:
            B_img = self.transform_B(B_img)
        return A_img, B_img

    def __len__(self):
        return max(self.A_len, self.B_len)

