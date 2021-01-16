# -*- coding: utf-8 -*-
# ======================================================
# @Time    : 21-1-16 下午1:30
# @Author  : huang ha
# @Email   : 1286304229@qq.com
# @File    : demo_mp.py
# @Comment: 
# ======================================================

from multiprocessing import Process
from multiprocessing import Pool
import os


def demo_1():
    def f(name1, name2):
        print('hello', name1, name2)
        print('子进程pid', os.getpid())

    p = Process(target=f, args=('andy', 'bob'))
    # 设置为守护进程
    p.daemon = True
    # 开始子进程
    p.start()
    # 主进程等待子进程执行完成，否则容易出现僵尸进程
    p.join()
    print('主进程pid', os.getpid())


def demo_2():
    class SubProcess(Process):
        def __init__(self, name1, name2):
            super().__init__()
            self.name1 = name1
            self.name2 = name2

        def run(self):
            print('hello', self.name1, self.name2)
            print('子进程pid', os.getpid())

    sub_process = SubProcess('andy', 'bob')
    # 开始子进程
    sub_process.start()
    # 主进程等待子进程执行完成，否则容易出现僵尸进程
    sub_process.join()
    print('主进程pid', os.getpid())


def demo_3():
    def f(name1, name2):
        print('hello', name1, name2)
        print('子进程pid', os.getpid())

    p = Process(target=f, args=('andy', 'bob'))
    # 设置为守护进程，一定要在p.start()前设置
    p.daemon = True
    # 开始子进程
    p.start()
    # 注释掉，模拟主进程退出，但是子进程还没有开始执行场景
    # p.join()
    print('主进程pid', os.getpid())


def demo_4():
    def f(x):
        print(x)
        return x * x

    with Pool(5) as p:
        p.map(f, [1, 2, 3])


def demo_5():
    def f(x):
        print(x)
        return x * x

    ps = []
    for i in range(1, 4):
        p = Process(target=f, args=(i,))
        ps.append(p)
    for p in ps:
        p.start()

    for p in ps:
        p.join()


def f(x, y):
    return x * x, y / 2


def demo_6():
    with Pool(5) as p:
        print(p.map(f, zip([1, 2, 3], [1, 2, 3])))


def f(args):
    return args[0] * args[0], args[1] / 2


def demo_7():
    with Pool(5) as p:
        print(p.map(f, ([1, 1], [2, 2], [3, 3])))


def f(x, y):
    return x * x, y / 2


def demo_8():
    with Pool(5) as p:
        print(p.starmap(f, zip([1, 2, 3], [1, 2, 3])))


def demo_9():
    from itertools import starmap

    def f(x, y):
        return x * x, y / 2

    # 报错TypeError: f() missing 1 required positional argument: 'y'
    # o = list(map(f, zip([1, 2, 3], [1, 2, 3])))
    o = list(starmap(f, zip([1, 2, 3], [1, 2, 3])))
    # 输出 [(1, 0.5), (4, 1.0), (9, 1.5)]
    print(o)


def demo_10():

    with Pool(5) as p:
        task = p.starmap_async(f, zip([1, 2, 3], [1, 2, 3]))
        print(task.get())


if __name__ == '__main__':
    # demo_1()
    # demo_2()
    # demo_3()
    # demo_4() # 无法运行会报错
    # demo_5()
    # 无法运行会报错
    # demo_6()
    # demo_7()
    # demo_8()
    # demo_9()
    demo_10()


