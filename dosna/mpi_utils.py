

import time
from math import ceil, log10
from mpi4py import MPI

from . import status


def mpi_comm():
    comm = status().engine.params.get('comm', None)
    if comm is None:
        comm = MPI.COMM_WORLD
    return comm


def mpi_rank():
    return mpi_comm().Get_rank()


def mpi_size():
    return mpi_comm().Get_size()


def mpi_barrier():
    return mpi_comm().Barrier()


def pprint(*args, **kwargs):
    rank = kwargs.pop('rank', None)
    pprint_prefix = '|{{0:{}d}})'.format(int(ceil(log10(mpi_size()))))
    if rank is None or rank == mpi_rank():
        print(pprint_prefix.format(mpi_rank()), *args, **kwargs) 


def mpi_root():
    return mpi_rank() == 0


class MpiRoot(object):
    
    def __enter__(self):
        if not mpi_root():
            raise Exception() # Forze early exit of rank != 0 proccesses
    
    def __exit__(self, *args):
        mpi_barrier()
        
        
class MpiTimer(object):
    def __init__(self, name='Timer'):
        self.name = name
        self.tstart = -1
        self.tend = -1
        self.time = 0

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        mpi_barrier()
        self.tend = time.time()
        self.time = self.tend - self.tstart
        pprint('%s -- Elapsed: %.4f seconds' % (self.name, self.time), rank=0)