# -*- coding: utf-8 -*-
# ======================================================
# @Time    : 21-1-16 下午10:24
# @Author  : huang ha
# @Email   : 1286304229@qq.com
# @File    : demo_pytorch_spawn.py
# @Comment : 需要cuda才能跑
# ======================================================

import torch
# import multiprocessing as mp
import torch.multiprocessing as mp
import numpy as np


class A(object):

    def __init__(self):
        # 如果设置为torch.ones(3).cuda() 那么必须开启spawn模式，否则程序直接报错
        self.b = torch.ones(3)


def f1(l):
    l.b *= 2
    print('子进程1：', l.b)


def f2(l):
    l.b *= 3
    print('子进程2：', l.b)


if __name__ == '__main__':
    mp.set_start_method('spawn')

    a = A()
    p1 = mp.Process(target=f1, args=(a,))
    p2 = mp.Process(target=f2, args=(a,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print('主进程：', a.b)

