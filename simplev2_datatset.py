# -*- coding: utf-8 -*-
# ======================================================
# @Time    : 20-12-25 下午9:49
# @Author  : huang ha
# @Email   :
# @File    : simple1_datatset.py
# @Comment: 
# ======================================================

from libv1 import Dataset
import numpy as np
import time


class SimpleV2Dataset(Dataset):
    def __init__(self):
        # 伪造数据
        self.imgs = np.arange(0, 160).reshape(80, 2)

    def __getitem__(self, index):
        # 模拟耗时操作
        time.sleep(0.5)
        return self.imgs[index]

    def __len__(self):
        return self.imgs.shape[0]
