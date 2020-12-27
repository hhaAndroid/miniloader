from torch.utils.data._utils.collate import default_collate
from .sampler import BatchSampler, SequentialSampler, RandomSampler


class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, collate_fn=None, drop_last=False):
        self.dataset = dataset

        # 因为这两个功能是冲突的，假设shuffle=True,但是sampler里面是SequentialSampler，那么就违背设计思想了
        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if batch_sampler is not None:
            # 一旦设置了batch_sampler，那么batch_size、shuffle、sampler和drop_last四个参数就不能传入
            # 因为这4个参数功能和batch_sampler功能冲突了
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            batch_size = None
            drop_last = False

        if sampler is None:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        # 也就是说batch_sampler必须要存在，你如果没有设置，那么采用默认类
        if batch_sampler is None:
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = iter(batch_sampler)

        if collate_fn is None:
            collate_fn = default_collate
        self.collate_fn = collate_fn

    # 核心
    def __next__(self):
        index = next(self.batch_sampler)
        data = [self.dataset[idx] for idx in index]
        data = self.collate_fn(data)
        return data

    # 返回自身，因为自身实现了 __next__
    def __iter__(self):
        return self
