# -*- coding: utf-8 -*-
# ======================================================
# @Time    : 20-12-26 下午4:42
# @Author  : huang ha
# @Email   :
# @File    : collate.py
# @Comment: 
# ======================================================
import torch


def default_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    elif elem_type.__module__ == 'numpy':
        return default_collate([torch.as_tensor(b) for b in batch])
    else:
        raise NotImplementedError
