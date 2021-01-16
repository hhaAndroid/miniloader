from multiprocessing import Process
import os
import random


class Demo(object):
    pass


class Demo21(Demo):
    def __init__(self):
        self.name = 'Demo21'
        self.rand = random.randint(1, 100)


class Demo22(Demo):
    def __init__(self):
        self.name = 'Demo22'
        self.rand = random.randint(1, 100)


class Demo31(Demo):
    def __init__(self):
        self.name = 'Demo21'
        self.rand = random.randint(1, 100)


def create_class(cls):
    if isinstance(cls, Demo):
        this = cls
    else:
        this = cls()

    print("ProcessID#{} InstanceID#{} this#{} name#{} rand#{}".format(os.getpid(), id(this), this, id(this.name),
                                                                      id(this.rand)))


if __name__ == '__main__':
    pool = []
    x = Demo21()
    for cls in [Demo21(), Demo21(), Demo31, Demo22, Demo22, x, x]:
        p = Process(target=create_class, args=(cls,))
        pool.append(p)
    for p in pool:
        p.start()
    for p in pool:
        p.join()
