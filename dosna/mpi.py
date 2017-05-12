

from __future__ import print_function

import numpy as np
import dosna as _dn
from mpi4py import MPI
from math import log10, ceil


MPI_COMM = MPI.COMM_WORLD
MPI_SIZE = MPI_COMM.Get_size()
MPI_RANK = MPI_COMM.Get_rank()


auto_init = _dn.connect
wait = Barrier = MPI_COMM.Barrier
rank = MPI_RANK
ncores = MPI_SIZE


pprint_prefix = '|{{0:{}d}})'.format(int(ceil(log10(MPI_SIZE))))
def pprint(*args, **kwargs):
    rank = kwargs.pop('rank', None)
    if rank is None or rank == MPI_RANK:
        print(pprint_prefix.format(MPI_RANK), *args, **kwargs)


class Cluster(_dn.Cluster):

    def create_pool(self, pool_name=None, auid=None, crush_rule=None, test=False, **kwargs):
        if MPI_RANK == 0:
            pool = super(Cluster, self).create_pool(pool_name=pool_name, aui=auid,
                                                    crush_rule=crush_rule,
                                                    test=test, **kwargs)
            pool_name = pool.name
        else:
            pool_name = None
        pool_name = MPI_COMM.bcast(pool_name, root=0)
        return self.get_pool(pool_name, **kwargs)

    def delete_pool(self, pool_name):
        MPI_COMM.Barrier()  # Wait for others
        if MPI_RANK == 0:
            super(Cluster, self).delete_pool(pool_name)
        MPI_COMM.Barrier()


class Pool(_dn.Pool):

    def __init__(self, name=None, **kwargs):
        if name is None:
            if MPI_RANK == 0:
                kwargs.setdefault('test', False)
                pname = Pool.random_name(test=kwargs['test'])
            else:
                pname = None
            pname = MPI_COMM.bcast(pname, root=0)
        else:
            pname = name
        super(Pool, self).__init__(name=pname, **kwargs)


    def create_dataset(self, name, shape=None, dtype=None, **kwargs):
        if MPI_RANK == 0:
            kwargs['autoload'] = False
            ds = super(Pool, self).create_dataset(name, shape=shape,
                                                  dtype=dtype, **kwargs)
            ds_name = ds.name
        else:
            ds_name = None
        ds_name = MPI_COMM.bcast(ds_name, root=0)
        ds = Dataset(self, ds_name, read_only=kwargs.get('readonly', False),
                     njobs=kwargs.get('njobs', self.njobs))

        if 'data' in kwargs and kwargs['data'] is not None:
            ds.load(kwargs['data'])
        return ds

    def delete_dataset(self, name):
        MPI_COMM.Barrier()
        super(Pool, self).delete_dataset(name)
        MPI_COMM.Barrier()

    def delete(self):
        MPI_COMM.Barrier()
        if MPI_RANK == 0:
            super(Pool, self).delete()
        MPI_COMM.Barrier()


class File(Pool):

    def __init__(self, name=None, open_mode='a', **kwargs):
        super(File, self).__init__(name=name, open_mode=open_mode, **kwargs)


class Dataset(_dn.Dataset):

    @classmethod
    def create(cls, pool, name, shape=None, dtype=np.float32, fillvalue=-1, chunks=None,
               data=None, read_only=False, njobs=None):
        if MPI_RANK == 0:
            dataset = super(Dataset, cls).create(pool, name, shape=shape, dtype=dtype,
                                                 fillvalue=fillvalue, chunks=chunks, data=data,
                                                 read_only=read_only, njobs=njobs)
            ds_name = dataset.name
        else:
            ds_name = None
        ds_name = MPI_COMM.bcast(ds_name, root=0)
        ds = cls(pool, ds_name, read_only=read_only, njobs=njobs or pool.njobs)
        if data is not None:
            ds.load(data)
        return ds

    def load(self, data, use_mpi=True):
        if use_mpi:
            for idx in range(MPI_RANK, self.total_chunks, MPI_SIZE):
                idx = self._transform_chunk_index(idx)
                gslices = self._gchunk_bounds_slices(idx)
                lslices = self._lchunk_bounds_slices(idx)
                self._set_chunk_data(idx, data[gslices], slices=lslices)
            MPI_COMM.Barrier()
        else:
            if MPI_RANK == 0:
                super(Dataset, self).load(data)
            MPI_COMM.Barrier()

    def clone(self, new_name):
        ds = Dataset.create(self.pool, new_name, shape=self.shape, dtype=self.dtype,
                            chunks=self.chunk_size, read_only=self.read_only,
                            njobs=self.njobs, fillvalue=self.fillvalue)
        MPI_COMM.Barrier()
        return ds

    def delete(self, delete_chunks=True):
        MPI_COMM.Barrier()
        if delete_chunks:
            for idx in range(MPI_RANK, self.total_chunks, MPI_SIZE):
                idx = self._transform_chunk_index(idx)
                self._delete_chunk(idx)
        if MPI_RANK == 0:
            super(Dataset, self).delete()
        MPI_COMM.Barrier()

    def map(self, new_name, func, *args, **kwargs):
        dsout = self.clone(new_name)
        for idx in range(MPI_RANK, self.total_chunks, MPI_SIZE):
            idx = self._transform_chunk_index(idx)
            slices = self._lchunk_bounds_slices(idx)
            result = func(self._get_chunk_data(idx, slices=slices), *args, **kwargs)
            dsout._set_chunk_data(idx, result, slices=slices)
        MPI_COMM.Barrier()
        return dsout

    def map_padded(self, out_name, func, padding, *args, **kwargs):
        pad_mode = kwargs.pop('mode', 'reflect')
        if type(padding) == int:
            padding = [padding] * self.ndim
        elif len(padding) != self.ndim:
            raise Exception('Padding does not match')

        dsout = self.clone(out_name)

        for idx in range(MPI_RANK, self.total_chunks, MPI_SIZE):
            idx = self._transform_chunk_index(idx)
            lS = self._lchunk_bounds_slices(idx) # local slices
            gS = self._gchunk_bounds_slices(idx) # global slices
            slices_in = []
            pad = []
            for i in range(self.ndim):
                pS = min(padding[i], gS[i].start)
                pE = min(padding[i], self.shape[i] - gS[i].stop)
                slices_in.append(slice(gS[i].start - pS, gS[i].stop + pE))
                pad.append((padding[i] - pS, padding[i] - pE))
            data = self[slices_in]
            data = np.pad(data, pad, pad_mode)
            kwargs['padding'] = tuple(padding)
            result = func(data, *args, **kwargs)
            dsout._set_chunk_data(idx, result, slices=lS)
        MPI_COMM.Barrier()
        return dsout


    def apply(self, func, *args, **kwargs):
        for idx in range(MPI_RANK, self.total_chunks, MPI_SIZE):
            idx = self._transform_chunk_index(idx)
            slices = self._lchunk_bounds_slices(idx)
            result = func(self._get_chunk_data(idx, slices=slices), *args, **kwargs)
            self._set_chunk_data(idx, result, slices=slices)
        MPI_COMM.Barrier()


class DataChunk(_dn.DataChunk):

    @classmethod
    def create(cls, pool, name, shape=None, dtype=None, fillvalue=None, data=None, slices=None):
        if MPI_RANK == 0:
            chunk = super(DataChunk, cls).create(pool, name, shape=shape, dtype=dtype,
                                                 fillvalue=fillvalue, data=data,
                                                 slices=slices)
            chk_name = chunk.name
        else:
            chk_name = None
        chk_name = MPI_COMM.bcast(chk_name, root=0)
        return cls(pool, chk_name)

    def delete(self):
        MPI_COMM.Barrier()
        if MPI_RANK == 0:
            super(DataChunk, self).delete()
        MPI_COMM.Barrier()