from argparse import ArgumentParser
import os
import torch
import torch.multiprocessing as mp
from torch import distributed as dist


def parse_args():
    parser = ArgumentParser(description="Torch Distributed Training Launcher")
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Number of nodes",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=3,
        help="Number of workers per node",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="node rank",
    )
    return parser.parse_args()


def main(functions, world_size=2, backend='gloo', start_method='fork', nnodes=1, node_rank=0):
    try:
        # start_method= 'fork' or 'spawn', default is spawn
        mp.start_processes(init_process,
                           args=(world_size, nnodes, node_rank, functions, backend),
                           nprocs=world_size,
                           start_method=start_method)
        # 如果启动方法是 spawn ，一般可以用简化写法
        # mp.spawn(
        #     init_process,
        #     args=(world_size, functions, backend),
        #     nprocs=world_size)
    except Exception:
        raise RuntimeError('----failed-----')


def init_process(local_rank, nproc_per_node, nnodes, node_rank, functions, backend='gloo'):
    """Initialize the distributed environment."""
    print(f'========node_rank: {node_rank} local_rank: {local_rank}=============')
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['WORLD_SIZE'] = str(nproc_per_node * nnodes)

    dist_rank = nproc_per_node * node_rank + local_rank
    os.environ["RANK"] = str(dist_rank)
    # 非必备,不影响程序运行，但是有用
    os.environ["LOCAL_RANK"] = str(local_rank)

    if backend == 'nccl':
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(local_rank % num_gpus)
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
        print('is_distributed:', dist.is_available() and dist.is_initialized())
        print('world_size', dist.get_world_size(group))
        print('backend', dist.get_backend(group))

    print(f'========rank:{dist.get_rank(group)}=======')
    print(f'========local rank:{os.environ["LOCAL_RANK"]}=========')


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
    args = parse_args()

    nnodes = args.nnodes  # 总共节点数，也就是总共有多少台机器
    nproc_per_node = args.nproc_per_node # 每个节点有多少进程
    node_rank = args.node_rank  # 当前是第几台机器

    functions = [print_info, all_gather]
    backend = 'gloo'

    main(functions, nproc_per_node, backend, 'fork', nnodes, node_rank)
