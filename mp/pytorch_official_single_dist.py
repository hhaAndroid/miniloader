import os
import torch
from torch import distributed as dist
import torch.multiprocessing as mp


# launch 启动脚本会通过 os.environ 形式设置环境变量
# 我们就可以通过读取了
def main(functions, backend='gloo'):
    """Initialize the distributed environment."""
    if backend == 'nccl':
        num_gpus = torch.cuda.device_count()
        # 应该用 LOCAL_RANK 才是最合理的
        # 大部分人图省事，用 RANK
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']) % num_gpus)
        device = 'cuda'
    else:
        device = 'cpu'

    dist.init_process_group(backend=backend)

    for func in functions:
        func(device)


def print_info(device, group=None):
    # master process
    if dist.get_rank() == 0:
        print(f'===========rank:{dist.get_rank(group)}============')
        print('device', device)
        print('start_method', mp.get_start_method())
        print('is_distributed:', dist.is_available() and dist.is_initialized())
        print('world_size', dist.get_world_size(group))
        print('backend', dist.get_backend(group))

    print(f'================rank:{dist.get_rank(group)}=================')


def all_gather(device, group=None):
    if dist.get_rank() == 0:
        data = torch.tensor([0, 1]).to(device)
    else:
        data = torch.tensor([1, 2]).to(device)

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    gather_list = [torch.empty_like(data) for _ in range(world_size)]
    dist.all_gather(gather_list, data, group)
    if dist.get_rank() == 0:
        print(gather_list)
    return gather_list


if __name__ == '__main__':
    main([print_info, all_gather],  'gloo')
