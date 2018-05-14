#!/usr/bin/env python
"""cpu """
import logging

import numpy as np

from dosna.backends import get_backend
from dosna.engines import Engine
from dosna.engines.base import EngineConnection, EngineDataChunk, EngineDataset
from six.moves import range

log = logging.getLogger(__name__)


class CpuConnection(EngineConnection):

    def __init__(self, *args, **kwargs):
        bname = kwargs.pop('backend', None)
        instance = get_backend(bname).Connection(*args, **kwargs)
        super(CpuConnection, self).__init__(instance)

    def get_dataset(self, name):
        dataset = self.instance.get_dataset(name)
        return CpuDataset(dataset)


class CpuDataset(EngineDataset):

    def get_chunk(self, idx):
        chunk = self.instance.get_chunk(idx)
        return CpuDataChunk(chunk)

    def get_data(self, slices=None):
        log.debug("getting data at %s[%s]", self.name, slices)
        slices, squeeze_axis = self._process_slices(slices, squeeze=True)
        tshape = tuple(x.stop - x.start for x in slices)
        chunk_iterator = self._chunk_slice_iterator(slices, self.ndim)

        output = np.empty(tshape, dtype=self.dtype)
        for idx, cslice, gslice in chunk_iterator:
            output[gslice] = self.get_chunk_data(idx, slices=cslice)

        if squeeze_axis:
            return np.squeeze(output, axis=squeeze_axis)
        return output

    def set_data(self, values, slices=None):
        log.debug("setting data at %s[%s]", self.name, slices)
        if slices is None:
            return self.load(values)

        isscalar = np.isscalar(values)
        ndim = self.ndim if isscalar else values.ndim
        slices, _ = self._process_slices(slices, squeeze=True)
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
        out = self.instance.connection.create_dataset(
            output_name, shape=self.shape,
            dtype=self.dtype, chunk_size=self.chunk_size,
            fillvalue=self.fillvalue)
        return CpuDataset(out)


class CpuDataChunk(EngineDataChunk):

    pass


# Export Engine
_engine = Engine('cpu', CpuConnection, CpuDataset, CpuDataChunk, {})
