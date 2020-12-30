from libv1 import DataLoader, default_collate


def demo_test_1():
    from simplev1_datatset import SimpleV1Dataset
    simple_dataset = SimpleV1Dataset()
    dataloader = DataLoader(simple_dataset, batch_size=2, collate_fn=default_collate)
    for data in dataloader:
        print(data)


def demo_test_2():
    from simplev1_datatset import SimpleV1Dataset
    from libv1 import SequentialSampler, RandomSampler
    from collections import Iterator, Iterable

    simple_dataset = SimpleV1Dataset()
    dataloader = DataLoader(simple_dataset, batch_size=2, collate_fn=default_collate)

    print(isinstance(simple_dataset, Iterable))
    print(isinstance(simple_dataset, Iterator))
    print(isinstance(iter(simple_dataset), Iterator))

    print(isinstance(SequentialSampler(simple_dataset), Iterable))
    print(isinstance(SequentialSampler(simple_dataset), Iterator))
    print(isinstance(iter(SequentialSampler(simple_dataset)), Iterator))

    # BatchSampler 和 RandomSampler 内部实现结构一样，结果也是一样
    print(isinstance(RandomSampler(simple_dataset), Iterable))
    print(isinstance(RandomSampler(simple_dataset), Iterator))
    print(isinstance(iter(RandomSampler(simple_dataset)), Iterator))

    print(isinstance(dataloader, Iterator))


def demo_test_3():
    class DataLoader(object):
        def __init__(self):
            self.dataset = [[img0, target0], [img1, target1], [img2, target2], ..., [img99, target99]]
            self.sampler = [0, 1, 2, 3, 4, ..., 99]
            self.batch_size = 4
            self.index = 0

        def collate_fn(self, data):
            batch_img = torch.stack(data[0], 0)
            batch_target = torch.stack(data[1], 0)
            return batch_img, batch_target

        def __next__(self):
            # 0.batch_index 输出
            i = 0
            batch_index = []
            while i < self.batch_size:
                batch_index.append(self.sampler[self.index])
                self.index += 1
                i += 1

            # 1.得到 batch 个数据了
            data = [self.dataset[idx] for idx in batch_index]

            # 2.collate_fn 在 batch 维度拼接输出
            batch_data = self.collate_fn(data)
            return batch_data

        def __iter__(self):
            return self


if __name__ == '__main__':
    demo_test_1()
    # demo_test_2()

