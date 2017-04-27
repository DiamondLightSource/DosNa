

import numpy as np
import dosna as _dn
from mpi4py import MPI


MPI_COMM = MPI.COMM_WORLD
MPI_SIZE = MPI_COMM.Get_size()
MPI_RANK = MPI_COMM.Get_rank()


auto_init = _dn.connect


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
        return cls(pool, name, read_only=read_only, njobs=njobs or pool.njobs)

    def load(self, data):
        for idx in range(MPI_RANK, self.total_chunks, MPI_SIZE):
            idx = self._transform_chunk_index(idx)
            gslices = self._gchunk_bounds_slices(idx)
            lslices = self._lchunk_bounds_slices(idx)
            self._set_chunk_data(idx, data[gslices], slices=lslices)
        MPI_COMM.Barrier()

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