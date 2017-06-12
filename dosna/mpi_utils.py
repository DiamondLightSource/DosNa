

from __future__ import with_statement, print_function
import sys
import inspect

import time
from math import ceil, log10
from mpi4py import MPI

from . import status


def mpi_comm():
    comm = status().engine.params.get('comm', None)
    if comm is None:
        comm = MPI.COMM_WORLD
    return comm


def mpi_rank(comm=None):
    comm = comm or mpi_comm()
    return comm.Get_rank()


def mpi_size(comm=None):
    comm = comm or mpi_comm()
    return comm.Get_size()


def mpi_barrier(comm=None):
    comm = comm or mpi_comm()
    return comm.Barrier()


def pprint(*args, **kwargs):
    rank = kwargs.pop('rank', None)
    comm = kwargs.pop('comm', None)
    pprint_prefix = '|{{0:{}d}})'.format(int(ceil(log10(mpi_size()))))
    if rank is None or rank == mpi_rank(comm=comm):
        print(pprint_prefix.format(mpi_rank()), *args, **kwargs)


def mpi_root(comm=None):
    return mpi_rank(comm=comm) == 0


class MpiTimer(object):
    def __init__(self, name='Timer', rank=0):
        self.name = name
        self.tstart = -1
        self.tend = -1
        self.time = 0
        self.rank = rank

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        mpi_barrier()
        self.tend = time.time()
        self.time = self.tend - self.tstart
        pprint('%s -- Elapsed: %.4f seconds' % (self.name, self.time),
               rank=self.rank)
