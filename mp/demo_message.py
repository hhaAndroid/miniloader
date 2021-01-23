# -*- coding: utf-8 -*-
# ======================================================
# @Time    : 21-1-16 下午1:30
# @Author  : huang ha
# @Email   : 1286304229@qq.com
# @File    : demo_message.py
# @Comment: 
# ======================================================

from multiprocessing import Process,Pipe,Queue
import multiprocessing as mp


def demo_1():
    def f(conn):
        # 发送
        conn.send([42, None, 'hello'])
        conn.close()

    # 默认是全双工，返回两个连接端点，一个发送一个接收，可以任意设置
    parent_conn, child_conn = Pipe()
    p = Process(target=f, args=(child_conn,))
    p.start()
    # 接收
    print(parent_conn.recv())  # prints "[42, None, 'hello']"
    # 等待子进程结束
    p.join()



def demo_2():
    def f(q):
        q.put([42, None, 'hello'])

    if __name__ == '__main__':
        q = Queue()
        p = Process(target=f, args=(q,))
        p.start()
        print(q.get())  # prints "[42, None, 'hello']"
        p.join()


if __name__ == '__main__':
    # mp.set_start_method('spawn')
    # demo_1()
    demo_2()



