from libv2 import DataLoader, default_collate

from torch.utils.data import DataLoader
import time


def demo_test_1():
    from simplev2_datatset import SimpleV2Dataset
    simple_dataset = SimpleV2Dataset()
    dataloader = DataLoader(simple_dataset, batch_size=2, collate_fn=default_collate)
    start = time.time()
    for data in dataloader:
        print(data)
    print('time(s):', time.time() - start)


def demo_test_2():
    from simplev2_datatset import SimpleV2Dataset
    simple_dataset = SimpleV2Dataset()
    dataloader = DataLoader(simple_dataset, batch_size=2, collate_fn=default_collate, num_workers=2)
    start = time.time()
    for data in dataloader:
        print(data)
    print('time(s):', time.time() - start)


if __name__ == '__main__':
    # demo_test_1()
    demo_test_2()
