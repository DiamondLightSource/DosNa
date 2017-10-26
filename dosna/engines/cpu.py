

import numpy as np

from .. import Engine
from ..base import Wrapper
from ..backends import get_backend

from six.moves import range

class CpuCluster(Wrapper):

    def __init__(self, *args, **kwargs):
        bname = kwargs.pop('backend', None)
        instance = get_backend(bname).Cluster(*args, **kwargs)
        super(CpuCluster, self).__init__(instance)

    def create_pool(self, *args, **kwargs):
        pool = self.instance.create_pool(*args, **kwargs)
        return CpuPool(pool)

    def get_pool(self, *args, **kwargs):
        pool = self.instance.get_pool(*args, **kwargs)
        return CpuPool(pool)

    def __getitem__(self, pool_name):
        return self.get_pool(pool_name)


class CpuPool(Wrapper):

    def create_dataset(self, *args, **kwargs):
        ds = self.instance.create_dataset(*args, **kwargs)
        ds = CpuDataset(ds)
        if 'data' in kwargs:
            ds.load(kwargs['data'])
        return ds

    def get_dataset(self, *args, **kwargs):
        ds = self.instance.get_dataset(*args, **kwargs)
        return CpuDataset(ds)

    def __getitem__(self, ds_name):
        return self.get_dataset(ds_name)


class CpuDataset(Wrapper):

    def create_chunk(self, *args, **kwargs):
        chunk = self.instance.create_chunk(*args, **kwargs)
        return CpuDataChunk(chunk)

    def get_chunk(self, *args, **kwargs):
        chunk = self.instance.get_chunk(*args, **kwargs)
        return CpuDataChunk(chunk)

    def __getitem__(self, slices):
        return self.get_data(slices)

    def __setitem__(self, slices, values):
        self.set_data(values, slices=slices)

    def get_data(self, slices=None):
        slices, squeeze_axis = self._process_slices(slices, squeeze=True)
        tshape = tuple(x.stop - x.start for x in slices)
        chunk_iterator = self._chunk_slice_iterator(slices, self.ndim)

        output = np.empty(tshape, dtype=self.dtype)
        for idx, cslice, gslice in chunk_iterator:
            output[gslice] = self.get_chunk_data(idx, slices=cslice)

        if len(squeeze_axis) > 0:
            return np.squeeze(output, axis=squeeze_axis)
        return output

    def set_data(self, values, slices=None):
        if slices is None:
            return self.load(values)

        isscalar = np.isscalar(values)
        ndim = self.ndim if isscalar else values.ndim
        slices, squeeze_axis = self._process_slices(slices, squeeze=True)
        chunk_iterator = self._chunk_slice_iterator(slices, ndim)

        for idx, cslice, gslice in chunk_iterator:
            if isscalar:
                self.set_chunk_data(idx, values, slices=cslice)
            else:
                self.set_chunk_data(idx, values[gslice], slices=cslice)

    def clear(self):
        for idx in range(self.total_chunks):
            idx = self._idx_from_flat(idx)
            self.del_chunk(idx)

    def delete(self):
        self.instance.pool.del_dataset(self.name)

    def load(self, data):
        if data.shape != self.shape:
            raise Exception('Data shape does not match')
        for idx in range(self.total_chunks):
            idx = self._idx_from_flat(idx)
            gslices = self._global_chunk_bounds(idx)
            lslices = self._local_chunk_bounds(idx)
            self.set_chunk_data(idx, data[gslices], slices=lslices)

    def map(self, func, output_name):
        out = self.clone(output_name)
        for idx in range(self.total_chunks):
            idx = self._idx_from_flat(idx)
            data = func(self.get_chunk_data(idx))
            out.set_chunk_data(idx, data)
        return out

    def apply(self, func):
        for idx in range(self.total_chunks):
            idx = self._idx_from_flat(idx)
            data = func(self.get_chunk_data(idx))
            self.set_chunk_data(idx, data)

    def clone(self, output_name):
        out = self.instance.pool.create_dataset(
            output_name, shape=self.shape,
            dtype=self.dtype, chunks=self.chunk_size,
            fillvalue=self.fillvalue)
        return CpuDataset(out)


class CpuDataChunk(Wrapper):

    pass


# Export Engine
__engine__ = Engine('cpu', CpuCluster, CpuPool, CpuDataset, CpuDataChunk, {})
