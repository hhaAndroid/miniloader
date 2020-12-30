import torch
import torch.multiprocessing as multiprocessing
import itertools
import os
from collections import namedtuple
import random
from torch.utils.data import _utils
from torch._utils import ExceptionWrapper
from torch._six import queue

from .fetch import _MapDatasetFetcher


class _BaseDataLoaderIter(object):
    def __init__(self, loader):
        self._dataset = loader.dataset
        self._drop_last = loader.drop_last
        self._index_sampler = loader.batch_sampler  # BatchSampler
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._persistent_workers = loader.persistent_workers
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = torch.empty((), dtype=torch.int64).random_().item()
        self._num_yielded = 0

    def __iter__(self):
        return self

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        raise NotImplementedError

    def __next__(self):
        data = self._next_data()
        self._num_yielded += 1
        return data

    next = __next__  # Python 2 compatibility

    def __len__(self) -> int:
        return len(self._index_sampler)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0
        self._dataset_fetcher = _MapDatasetFetcher(loader.dataset, loader.collate_fn)

    # 迭代核心函数，返回的已经是 batch data 了
    def _next_data(self):
        # 输出 batch 个 index
        index = self._next_index()  # may raise StopIteration
        # 迭代 dataset+collate_fn
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        return data


class ManagerWatchdog(object):
    def __init__(self):
        self.manager_pid = os.getppid()
        self.manager_dead = False

    def is_alive(self):
        if not self.manager_dead:
            self.manager_dead = os.getppid() != self.manager_pid
        return not self.manager_dead


# 核心类
class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_MultiProcessingDataLoaderIter, self).__init__(loader)
        assert self._num_workers > 0
        # 预取 batch 长度,默认是2，如果 num_worker太多，则开始缓存的 batch 数据会很大，导致 OOM
        assert self._prefetch_factor > 0
        self._shutdown = False
        # 默认采用 torch 的多进程
        multiprocessing_context = multiprocessing
        self._worker_init_fn = loader.worker_init_fn  # 每个 worker 的初始化函数
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))  # 循环输出
        # 每个 worker 会将所处理的 batch data 推送到这个全局共享队列中
        self._worker_result_queue = multiprocessing_context.Queue()
        # 通信机制
        self._workers_done_event = multiprocessing_context.Event()

        # 每个 worker 都有唯一的 index 队列，用于接收该 worker 要处理的 batch index
        self._index_queues = []
        # 进程相关信息存储
        self._workers = []

        for i in range(self._num_workers):
            # No certainty which module multiprocessing_context is
            index_queue = multiprocessing_context.Queue()  # type: ignore
            # index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=_worker_loop,
                args=(self._dataset, index_queue,
                      self._worker_result_queue, self._workers_done_event,
                      self._collate_fn,
                      self._base_seed + i, self._worker_init_fn, i))
            # 设置为守护进程
            w.daemon = True
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)

        self._data_queue = self._worker_result_queue
        self._reset(loader, first_iter=True)

    def _reset(self, loader, first_iter=False):
        super()._reset(loader, first_iter)
        # 主进程已经发送了多少个batch的index(仅仅发送即可，不需要考虑worker是否处理完成)，发送一个累加一次
        self._send_idx = 0
        # 主进程已经完成接收多少个batch的data，接收一个累加一次，才能保证输出顺序
        self._rcvd_idx = 0
        # 主线程中任务信息，可能包括两种格式数据(worker_id,)、(worker_id, data)
        # (worker_id,)表示data数据还没有获取，例如 主进程发送一次batch indx到对应worker_id后，就会存储一个(worker_id,)
        # (worker_id, data)表示数据已经获取到了，马上就可以发出去
        # 所以这个变量非常重要
        self._task_info = {}
        # count(v for v in _task_info.values() if len(v) == 1) 存储当前维持多少个还没有彻底处理的
        # 该参数可以反映拥堵程度
        self._tasks_outstanding = 0
        # 存储每个worker的工作状态，如果是 map迭代模式，其实始终为True
        self._workers_status = [True for i in range(self._num_workers)]
        if not first_iter:
            # 如果希望每个epoch直接的worker进程不销毁，可以采用如下模式
            for idx in range(self._num_workers):
                # 特意插入特定类型数据
                self._index_queues[idx].put(_ResumeIteration())
            resume_iteration_cnt = self._num_workers
            while resume_iteration_cnt > 0:
                # 主进程获取数据
                data = self._get_data()
                if isinstance(data, _ResumeIteration):
                    resume_iteration_cnt -= 1
        # 主进程预先推送 self._prefetch_factor * self._num_workers 个 batch index,也就是生产者先生产一批数据，可以加快初始迭代
        for _ in range(self._prefetch_factor * self._num_workers):
            # 主进程发送数据
            self._try_put_index()

    # 主进程从data_queue中获取数据
    def _get_data(self):
        # Fetches data from `self._data_queue`.
        #
        # We check workers' status every `MP_STATUS_CHECK_INTERVAL` seconds,
        # which we achieve by running `self._try_get_data(timeout=MP_STATUS_CHECK_INTERVAL)`
        # in a loop. This is the only mechanism to detect worker failures for
        # Windows. For other platforms, a SIGCHLD handler is also used for
        # worker failure detection.
        #
        # If `pin_memory=True`, we also need check if `pin_memory_thread` had
        # died at timeouts.
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self._timeout))
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def _try_get_data(self, timeout=5.0):
        # Tries to fetch data from `self._data_queue` once for a given timeout.
        # This can also be used as inner loop of fetching without timeout, with
        # the sender status as the loop condition.
        #
        # This raises a `RuntimeError` if any worker died expectedly. This error
        # can come from either the SIGCHLD handler in `_utils/signal_handling.py`
        # (only for non-Windows platforms), or the manual check below on errors
        # and timeouts.
        #
        # Returns a 2-tuple:
        #   (bool: whether successfully get data, any: data if successful else None)
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            # 主现场获取数据有异常
            # At timeout and error, we manually check whether any worker has
            # failed. Note that this is the only mechanism for Windows to detect
            # worker failures.
            failed_workers = []
            # 检查异常线程
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)
            # 如果有线程异常，那么就只能退出了，因为此时无法再保证顺序
            if len(failed_workers) > 0:
                pids_str = ', '.join(str(w.pid) for w in failed_workers)
                raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) from e
            # 如果是队列空异常
            if isinstance(e, queue.Empty):
                return (False, None)

    # 主进程通知对应子进程
    def _mark_worker_as_unavailable(self, worker_id, shutdown=False):
        # Mark a worker as having finished its work e.g., due to
        # exhausting an `IterableDataset`. This should be used only when this
        # `_MultiProcessingDataLoaderIter` is going to continue running.

        assert self._workers_status[worker_id] or (self._persistent_workers and shutdown)

        # Signal termination to that specific worker.
        q = self._index_queues[worker_id]
        # Indicate that no more data will be put on this queue by the current
        # process.
        q.put(None)  # 下一个推送 None ，然后该 worker 就会被销毁

        # Note that we don't actually join the worker here, nor do we remove the
        # worker's pid from C side struct because (1) joining may be slow, and
        # (2) since we don't join, the worker may still raise error, and we
        # prefer capturing those, rather than ignoring them, even though they
        # are raised after the worker has finished its job.
        # Joinning is deferred to `_shutdown_workers`, which it is called when
        # all workers finish their jobs (e.g., `IterableDataset` replicas) or
        # when this iterator is garbage collected.

        self._workers_status[worker_id] = False  # 设置为进程不可用

        assert self._workers_done_event.is_set() == shutdown

    # 主进程推送 batch index 到 worker 进程
    def _try_put_index(self):
        assert self._tasks_outstanding < self._prefetch_factor * self._num_workers

        try:
            # BatchSampler 进行迭代采样，和单进程操作完全相同，返回 batch index
            index = self._next_index()
        except StopIteration:
            return
        # 循环遍历 worker 进程，只要是没有被销毁，那就给你发
        for _ in range(self._num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        # 推送 batch index 索引，和对应的 index，这两个字段都非常重要，缺一不可
        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        # 存储任务信息
        self._task_info[self._send_idx] = (worker_queue_idx,)
        # 未完成任务个数+1
        self._tasks_outstanding += 1
        # 发送 index个数 +1
        self._send_idx += 1

    # 主进程迭代获取数据输出
    def _next_data(self):
        while True:
            # If the worker responsible for `self._rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # we try to advance `self._rcvd_idx` to find the next valid index.
            #
            # This part needs to run in the loop because both the `self._get_data()`
            # call and `_IterableDatasetStopIteration` check below can mark
            # extra worker(s) as dead.
            # 正常情况下，self._rcvd_idx < self._send_idx
            # 并且 self._workers_status[worker_id]=True
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[worker_id]:  # has data or is still active
                    break
                # 异常情况，例如某个进程异常，此时就只能忍痛删掉该进程要处理的任务，并且跳过这个任务
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # 这种情况表示
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                # 取数据
                data = self._task_info.pop(self._rcvd_idx)[1]
                # 返回数据，并发送一次数据
                return self._process_data(data)

            assert not self._shutdown and self._tasks_outstanding > 0
            # 主进程从data_queue中获取数据
            idx, data = self._get_data()
            # 未完成任务减1
            self._tasks_outstanding -= 1

            # 说明不是当前要返回的数据，只能先暂存
            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                # 否则直接返回，并发送下一个
                del self._task_info[idx]
                return self._process_data(data)

    def _process_data(self, data):
        # 成功获取一次数据，+1
        self._rcvd_idx += 1
        # 同时主进程再次插入任务，然后就可以构成循环，主进程发送一次，主进程获取一次，worker 同时运行
        self._try_put_index()
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        return data

    # 安全关闭所有任务，退出
    def _shutdown_workers(self):
        # Called when shutting down this `_MultiProcessingDataLoaderIter`.
        # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
        # the logic of this function.
        python_exit_status = _utils.python_exit_status
        if python_exit_status is True or python_exit_status is None:
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self._shutdown:
            self._shutdown = True
            try:
                # Exit `pin_memory_thread` first because exiting workers may leave
                # corrupted data in `worker_result_queue` which `pin_memory_thread`
                # reads from.
                if hasattr(self, '_pin_memory_thread'):
                    # Use hasattr in case error happens before we set the attribute.
                    self._pin_memory_thread_done_event.set()
                    # Send something to pin_memory_thread in case it is waiting
                    # so that it can wake up and check `pin_memory_thread_done_event`
                    self._worker_result_queue.put((None, None))
                    self._pin_memory_thread.join()
                    self._worker_result_queue.cancel_join_thread()
                    self._worker_result_queue.close()

                # Exit workers now.
                self._workers_done_event.set()  # 所有子进程都要退出
                for worker_id in range(len(self._workers)):
                    # Get number of workers from `len(self._workers)` instead of
                    # `self._num_workers` in case we error before starting all
                    # workers.
                    # If we are using workers_status with persistent_workers
                    # we have to shut it down because the worker is paused
                    if self._persistent_workers or self._workers_status[worker_id]:
                        self._mark_worker_as_unavailable(worker_id, shutdown=True)
                for w in self._workers:
                    w.join(timeout=5)
                    if w.is_alive():
                        # Existing mechanisms try to make the workers exit
                        # peacefully, but in case that we unfortunately reach
                        # here, which we shouldn't, (e.g., pytorch/pytorch#39570),
                        # we kill the worker.
                        w.terminate()
                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()

            finally:
                # Even though all this function does is putting into queues that
                # we have called `cancel_join_thread` on, weird things can
                # happen when a worker is killed by a signal, e.g., hanging in
                # `Event.set()`. So we need to guard this with SIGCHLD handler,
                # and remove pids from the C side data structure only at the
                # end.
                #
                # FIXME: Unfortunately, for Windows, we are missing a worker
                #        error detection mechanism here in this function, as it
                #        doesn't provide a SIGCHLD handler.
                # if self._worker_pids_set:
                #     _utils.signal_handling._remove_worker_pids(id(self))
                #     self._worker_pids_set = False
                pass

    def __del__(self):
        self._shutdown_workers()


r"""Dummy class used to resume the fetching when worker reuse is enabled"""
_ResumeIteration = namedtuple('_ResumeIteration', [])

# 每个 worker 循环取 batch index，并且通过 dataset+collate_fn，输入batch data 到 全局的 data_queue 中
def _worker_loop(dataset, index_queue, data_queue, done_event,
                 collate_fn, seed, init_fn, worker_id):
    try:
        # Initialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
        # module's handlers are executed after Python returns from C low-level
        # handlers, likely when the same fatal signal had already happened
        # again.
        # https://docs.python.org/3/library/signal.html#execution-of-python-signal-handlers

        # signal_handling._set_worker_signal_handlers()

        torch.set_num_threads(1)
        random.seed(seed)
        torch.manual_seed(seed)

        # global _worker_info
        # _worker_info = WorkerInfo(id=worker_id, num_workers=num_workers,
        #                           seed=seed, dataset=dataset)

        init_exception = None

        try:
            if init_fn is not None:
                init_fn(worker_id)

            fetcher = _MapDatasetFetcher(dataset, collate_fn)
        except Exception:
            init_exception = ExceptionWrapper(
                where="in DataLoader worker process {}".format(worker_id))

        iteration_end = False

        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                # 尝试获取 batch index
                r = index_queue.get(timeout=5.0)
            except queue.Empty:  # 队列空，则继续循环，直到有数据
                continue
            # 如果外面插入了 _ResumeIteration，表示该epoch worker运行完成，也不要退出，复用
            if isinstance(r, _ResumeIteration):
                # Acknowledge the main process
                data_queue.put(r)
                iteration_end = False
                # Recreate the fetcher for worker-reuse policy
                fetcher = _MapDatasetFetcher(dataset, collate_fn)
                continue
            # 接收到 None，表示 该进程出现了某种异常，所以该进程就要销毁
            elif r is None:
                # 接收到完成训练，可以终止该 worker 进程
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                # 跳过当前
                # `done_event` is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue
            idx, index = r
            # 如果出现异常，那么当前数据直接返回
            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    # dtaset+collate_fn 进行组成 Batch data
                    data = fetcher.fetch(index)
                except Exception as e:
                    data = ExceptionWrapper(
                        where="in DataLoader worker process {}".format(worker_id))
            data_queue.put((idx, data))  # 插入共享队列
            del data, idx, index, r  # save memory
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    # 主进程发送了退出命令，取消
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()



