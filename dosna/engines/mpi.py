#!/usr/bin/env python
"""Backend cpu uses the current computer cpu to run the main operations on
datasets"""

import logging

import numpy as np
from mpi4py import MPI

from dosna.backends import get_backend
from dosna.engines import Engine
from dosna.engines.base import EngineConnection, EngineDataChunk
from dosna.engines.cpu import CpuDataset
from six.moves import range

log = logging.getLogger(__name__)


class MpiMixin(object):

    def __init__(self, mpi_comm):
        self.mpi_comm = mpi_comm
        self.mpi_size = self.mpi_comm.Get_size()
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_is_root = self.mpi_rank == 0

    def mpi_barrier(self):
        self.mpi_comm.Barrier()


class MpiConnection(EngineConnection, MpiMixin):

    def __init__(self, name, comm=None, **kwargs):
        bname = kwargs.pop('backend', None)
        backend = get_backend(bname)

        bcomm = comm or _engine.params['comm']
        self.cparams = backend, name, kwargs
        self.ceph_backend = (backend.name == 'ceph')
        instance = backend.Connection(name, **kwargs)

        EngineConnection.__init__(self, instance)
        MpiMixin.__init__(self, bcomm)

        if backend.name == 'memory':
            log.warning('MPI engine will work unexpectedly with RAM backend')

    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunks=None):

        if self.mpi_is_root:
            self.instance.create_dataset(name, shape, dtype, fillvalue, data,
                                         chunks)
        self.mpi_barrier()
        dataset = self.get_dataset(name)
        if data is not None:
            dataset.load(data)
        return dataset

    def get_dataset(self, name):
        dataset = self.instance.get_dataset(name)
        return MpiDataset(dataset, self.mpi_comm)

    def del_dataset(self, name):
        dataset = self.get_dataset(name)
        dataset.clear()
        if self.mpi_is_root:
            self.instance.del_dataset(name)
        self.mpi_barrier()

    def __getitem__(self, ds_name):
        return self.get_dataset(ds_name)


class MpiDataset(CpuDataset, MpiMixin):

    def __init__(self, ds, mpi_comm):
        CpuDataset.__init__(self, ds)
        MpiMixin.__init__(self, mpi_comm)

    def create_chunk(self, idx, data=None, slices=None):
        if self.mpi_is_root:
            self.instance.create_chunk(idx, data, slices)
        self.mpi_barrier()
        return self.get_chunk(idx)

    def get_chunk(self, idx):
        chunk = self.instance.get_chunk(idx)
        return MpiDataChunk(chunk, self)

    def clear(self):
        for idx in range(self.mpi_rank, self.total_chunks, self.mpi_size):
            idx = self._idx_from_flat(idx)
            self.del_chunk(idx)
        self.mpi_barrier()

    def delete(self):
        self.clear()
        if self.mpi_is_root:
            self.instance.connection.del_dataset(self.name)
        self.mpi_barrier()

    def load(self, data):
        if data.shape != self.shape:
            raise Exception('Data shape does not match')
        for idx in range(self.mpi_rank, self.total_chunks, self.mpi_size):
            idx = self._idx_from_flat(idx)

            gslices = self._global_chunk_bounds(idx)
            lslices = self._local_chunk_bounds(idx)
            self.set_chunk_data(idx, data[gslices], slices=lslices)
        self.mpi_barrier()

    def map(self, func, output_name):
        out = self.clone(output_name)
        for idx in range(self.mpi_rank, self.total_chunks, self.mpi_size):
            idx = self._idx_from_flat(idx)
            data = func(self.get_chunk_data(idx))
            out.set_chunk_data(idx, data)
        self.mpi_barrier()
        return out

    def apply(self, func):
        for idx in range(self.mpi_rank, self.total_chunks, self.mpi_size):
            idx = self._idx_from_flat(idx)
            data = func(self.get_chunk_data(idx))
            self.set_chunk_data(idx, data)
        self.mpi_barrier()

    def clone(self, output_name):
        if self.mpi_is_root:
            out = self.instance.connection.create_dataset(
                output_name, shape=self.shape,
                dtype=self.dtype, chunks=self.chunk_size,
                fillvalue=self.fillvalue)
        self.mpi_barrier()
        out = self.instance.connection.get_dataset(output_name)
        return MpiDataset(out, self.mpi_comm)


class MpiDataChunk(EngineDataChunk):
    pass


# Export Engine
_engine = Engine('mpi', MpiConnection, MpiDataset, MpiDataChunk,
                 dict(comm=MPI.COMM_WORLD))
