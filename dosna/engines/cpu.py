#!/usr/bin/env python
"""cpu """
import logging

import numpy as np

from dosna.backends import get_backend
from dosna.engines import Engine
from dosna.engines.base import EngineConnection, EngineGroup, EngineLink, EngineDataChunk, EngineDataset
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
    
    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunk_size=None):
        dataset = self.instance.create_dataset(name, shape, dtype, fillvalue,
                                               data, chunk_size)
        engine_dataset = CpuDataset(dataset)
        if data is not None:
            engine_dataset.load(data)
        return engine_dataset
    
    def get_group(self, name):
        group = self.instance.get_group(name)
        return CpuGroup(group)

    def create_group(self, name, attrs={}):
        group = self.instance.create_group(name,attrs)
        engine_group = CpuGroup(group)
        return engine_group


    
class CpuGroup(EngineGroup):
    
    def create_group(self, name, attrs={}):
        group = self.instance.create_group(name, attrs)
        engine_group = CpuGroup(group)
        return engine_group
    
    def get_group(self, name):
        group = self.instance.get_group(name)
        return CpuGroup(group)

    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunk_size=None):
        dataset = self.instance.create_dataset(name, shape, dtype, fillvalue,
                                               data, chunk_size)
        engine_dataset = CpuDataset(dataset)
        if data is not None:
            engine_dataset.load(data)
        return engine_dataset
    
    def get_dataset(self, name):
        dataset = self.instance.get_dataset(name)
        return CpuDataset(dataset)

    def get_object(self, name):
        object = self.instance.get_object(name)
        # TODO differentiate group from dataset
        return object
    
class CpuLink(EngineLink):
    
    def get_source(self):
        return self.instance.get_source()
    
    def get_target(self):
        return self.instance.get_target()
    
    def get_name(self):
        return self.instance.get_name()


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
