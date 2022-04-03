import torch.multiprocessing as mp
import os
import torch
from torch import distributed as dist

def main(functions, world_size=2, backend='gloo', start_method='fork'):
    try:
        # start_method= 'fork' or 'spawn', default is spawn
        mp.start_processes(init_process,
                           args=(world_size, functions, backend),
                           nprocs=world_size,
                           start_method=start_method)
        # 如果启动方法是 spawn ，一般可以用简化写法
        # mp.spawn(
        #     init_process,
        #     args=(world_size, functions, backend),
        #     nprocs=world_size)
    except Exception:
        raise RuntimeError('----failed-----')


def init_process(rank, world_size, functions, backend='gloo'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29505'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    if backend == 'nccl':
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        device = 'cuda'
    else:
        device = 'cpu'

    dist.init_process_group(backend=backend)

    for func in functions:
        func(device)


# import torch.distributed.launch
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
    functions = [print_info, all_gather]
    world_size = 3
    backend = 'gloo'

    main(functions, world_size, backend)
