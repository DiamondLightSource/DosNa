

import logging as log
import numpy as np

import tempfile
from joblib import Parallel, delayed, dump, load

from .. import Engine
from ..base import Wrapper
from ..backends import get_backend
from .cpu import CpuDataset

from six.moves import range


class JoblibCluster(Wrapper):

    def __init__(self, *args, **kwargs):
        bname = kwargs.pop('backend', None)
        backend = get_backend(bname)
        instance = backend.Cluster(*args, **kwargs)

        super(JoblibCluster, self).__init__(instance)
        self.njobs = kwargs.pop('njobs', None) or __engine__.params['njobs']
        self.jlbackend = kwargs.pop('jlbackend', None) \
                         or __engine__.params['backend']

        if backend.name == 'memory' and __engine__.params['backend'] == 'multiprocessing':
            log.warning('Joblib engine will work unexpectedly with Memory backend')

    def create_pool(self, *args, **kwargs):
        pool = self.instance.create_pool(*args, **kwargs)
        return JoblibPool(pool, self.njobs, self.jlbackend)

    def get_pool(self, *args, **kwargs):
        pool = self.instance.get_pool(*args, **kwargs)
        return JoblibPool(pool, self.njobs, self.jlbackend)

    def __getitem__(self, pool_name):
        return self.get_pool(pool_name)


class JoblibPool(Wrapper):

    def __init__(self, pool, njobs, jlbackend):
        super(JoblibPool, self).__init__(pool)
        self.njobs = njobs
        self.jlbackend = jlbackend

    def create_dataset(self, *args, **kwargs):
        ds = self.instance.create_dataset(*args, **kwargs)
        ds = JoblibDataset(ds, self.njobs, self.jlbackend)
        if 'data' in kwargs:
            ds.load(kwargs['data'])
        return ds

    def get_dataset(self, *args, **kwargs):
        ds = self.instance.get_dataset(*args, **kwargs)
        return JoblibDataset(ds, self.njobs, self.jlbackend)

    def __getitem__(self, ds_name):
        return self.get_dataset(ds_name)


class JoblibDataset(CpuDataset):

    def __init__(self, ds, njobs, jlbackend):
        super(JoblibDataset, self).__init__(ds)
        self.njobs = njobs
        self.jlbackend = jlbackend

    def create_chunk(self, *args, **kwargs):
        chunk = self.instance.create_chunk(*args, **kwargs)
        return JoblibDataChunk(chunk)

    def get_chunk(self, *args, **kwargs):
        chunk = self.instance.get_chunk(*args, **kwargs)
        return JoblibDataChunk(chunk)

    def _make_temporary_memmap(self, filename, data=None, shape=None):
        if data is not None:
            dump(data, filename)
            return load(filename, mmap_mode='r')
        elif shape is not None:
            return np.memmap(filename, dtype=self.dtype, shape=shape, mode='w+')
        else:
            raise ValueError('Incorrect shape or values for memmapping')

    def get_data(self, slices=None):
        slices, squeeze_axis = self._process_slices(slices, squeeze=True)
        tshape = tuple(x.stop - x.start for x in slices)
        chunk_iterator = self._chunk_slice_iterator(slices, self.ndim)

        with tempfile.NamedTemporaryFile() as f:
            output = self._make_temporary_memmap(f.name, shape=tshape)
            Parallel(n_jobs=self.njobs, backend=self.jlbackend)(
                     delayed(_get_chunk_data_joblib)(self, idx, cslice, gslice, output)
                     for idx, cslice, gslice in chunk_iterator
            )
            output = np.asarray(output)

        if len(squeeze_axis) > 0:
            return np.squeeze(output, axis=squeeze_axis)
        return output

    def set_data(self, values, slices=None):
        if slices is None:
            return self.load(values)
        ndim = self.ndim if np.isscalar(values) else values.ndim
        slices, squeeze_axis = self._process_slices(slices, squeeze=True)
        chunk_iterator = self._chunk_slice_iterator(slices, ndim)

        Parallel(n_jobs=self.njobs, backend=self.jlbackend)(
            delayed(_set_chunk_data_joblib)(self, idx, cslice, gslice, values)
            for idx, cslice, gslice in chunk_iterator
        )

    def clear(self):
        Parallel(n_jobs=self.njobs, backend=self.jlbackend)(
            delayed(_delete_chunk)(self, idx)
            for idx in range(self.total_chunks)
        )

    def delete(self):
        self.instance._pool.del_dataset(self.name)

    def load(self, data):
        if data.shape != self.shape:
            raise Exception('Data shape does not match')

        with tempfile.NamedTemporaryFile() as f:
            dinput = self._make_temporary_memmap(f.name, data=data)
            Parallel(n_jobs=self.njobs, backend=self.jlbackend)(
                delayed(_populate_dataset_joblib)(self, idx, dinput)
                for idx in range(self.total_chunks)
            )

    def map(self, func, output_name):
        out = self.clone(output_name)
        Parallel(n_jobs=self.njobs, backend=self.jlbackend)(
            delayed(_map_function_chunk)(self, idx, out, func)
            for idx in range(self.total_chunks)
        )
        return out

    def apply(self, func):
        Parallel(n_jobs=self.njobs, backend=self.jlbackend)(
            delayed(_apply_function_chunk)(self, idx, func)
            for idx in range(self.total_chunks)
        )

    def clone(self, output_name):
        out = self.instance._pool.create_dataset(
            output_name, shape=self.shape,
            dtype=self.dtype, chunks=self.chunk_size,
            fillvalue=self.fillvalue)
        return JoblibDataset(out)


def _populate_dataset_joblib(inst, idx, dinput):
    idx = inst._idx_from_flat(idx)
    gslices = inst._global_chunk_bounds(idx)
    cslices = inst._local_chunk_bounds(idx)
    inst.set_chunk_data(idx, dinput[gslices], slices=cslices)

def _delete_chunk(inst, idx):
    inst.del_chunk(inst._idx_from_flat(idx))

def _map_function_chunk(inst, idx, out, func):
    idx = inst._idx_from_flat(idx)
    data = func(inst.get_chunk_data(idx))
    out.set_chunk_data(idx, data)

def _apply_function_chunk(inst, idx, func):
    idx = inst._idx_from_flat(idx)
    data = func(inst.get_chunk_data(idx))
    inst.set_chunk_data(idx, data)

def _get_chunk_data_joblib(inst, chunk_idx, cslice, gslice, doutput):
    doutput[gslice] = inst.get_chunk_data(chunk_idx, slices=cslice)

def _set_chunk_data_joblib(inst, chunk_idx, cslice, gslice, dinput):
    if np.isscalar(dinput):
        inst.set_chunk_data(chunk_idx, dinput, slices=cslice)
    else:
        inst.set_chunk_data(chunk_idx, dinput[gslice], slices=cslice)


class JoblibDataChunk(Wrapper):

    pass


# Export Engine
__engine__ = Engine('joblib', JoblibCluster, JoblibPool, JoblibDataset,
                    JoblibDataChunk, dict(njobs=-1, backend='multiprocessing'))
