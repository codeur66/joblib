# -*- coding: utf-8 -*-
# @Author: Thomas Moreau
# @Date:   2017-03-09 15:08:57
# @Last Modified by:   Thomas Moreau
# @Last Modified time: 2017-03-09 17:51:05
import random
from ._memmapping_reducer import get_memmapping_reducers, delete_folder
from loky.reusable_executor import get_reusable_executor


def get_memmapping_executor(n_jobs, **backend_args):
    """Factory for ReusableExecutor with automatix memmapping for large numpy
    arrays.
    """

    id_executor = random.randint(0, int(1e10))
    job_reducers, result_reducers, tmp_folder = get_memmapping_reducers(
        id_executor, **backend_args)
    _executor = get_reusable_executor(n_jobs, job_reducers=job_reducers,
                                      result_reducers=result_reducers)
    _executor._temp_folder = tmp_folder
    return _executor


class MemmappingExecutor():
    """Wrapper around ReusableExecutor to ease memmapping testing with Pool
    and Executor. This is only for testing purposes.
    """
    def __init__(self, n_jobs, **backend_args):
        self._executor = get_memmapping_executor(n_jobs, **backend_args)
        self._temp_folder = self._executor._temp_folder

    def apply_async(self, func, args):
        """Schedule a func to be run"""
        future = self._executor.submit(func, *args)
        future.get = future.result
        return future

    def terminate(self):
        self._executor.shutdown()
        delete_folder(self._executor._temp_folder)

    def map(self, f, *args):
        res = self._executor.map(f, *args)
        return list(res)
