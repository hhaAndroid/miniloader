# -*- coding: utf-8 -*-
# ======================================================
# @Time    : 21-1-16 上午11:56
# @Author  : huang ha
# @Email   : 1286304229@qq.com
# @File    : demo_spawn.py
# @Comment: 
# ======================================================

from multiprocessing import Process
import multiprocessing as mp
import numpy as np


class A(object):

    def __init__(self):
        self.b = np.ones(3)
        self.fn = lambda f: f * 2

    # 对象序列化时候调用
    def __getstate__(self):
        print('getstate')
        # 这里的fn，我直接将函数变成str进行序列化了
        return self.b, 'lambda f: f * 2'

    # 对象反序列化时候调用
    def __setstate__(self, state):
        print('setstate')
        super(A, self).__setattr__('b', state[0])
        super(A, self).__setattr__('fn', state[1])


def f1(l):
    l.b *= 2
    print('子进程1：', l.b)
    # 由于序列化原因，这里的fn是str
    print('子进程1：', l.fn)


def f2(l):
    l.b *= 3
    print('子进程2：', l.b)
    # 可以直接调用eval()还原为函数
    print('子进程2：', eval(l.fn))


if __name__ == '__main__':
    mp.set_start_method('spawn')

    a = A()
    p1 = Process(target=f1, args=(a,))
    p2 = Process(target=f2, args=(a,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print('主进程：', a.b)
    print('主进程：', a.fn)
