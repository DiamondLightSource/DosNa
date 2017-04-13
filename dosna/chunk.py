

import numpy as np

from .base import BaseData
from .utils import shape2str, dtype2str


class DataChunkException(Exception):
    pass


class DataChunk(BaseData):

    def __init__(self, pool, name, read_only=False):
        super(DataChunk, self).__init__(pool, name, read_only)
        if not pool.has_chunk(name):
            raise DataChunkException('No chunk `{}` found on pool `{}`'
                                     .format(name, pool.name))

    ###########################################################
    # DATA READING/WRITING
    ###########################################################

    def __getitem__(self, slices):
        return self.get_slices(slices)

    def __setitem__(self, slices, value=-1):
        self.set_slices(slices, value)

    def get_data(self):
        n = np.prod(self.shape)
        data = np.fromstring(self.read(), dtype=self.dtype, count=n)
        data.shape = self.shape  # in-place reshape
        return data

    def set_data(self, data):
        if data.shape != self.shape:
            raise DataChunkException('Cannot set chunk of shape `{}` with data of shape `{}`'
                                   .format(self.shape, data.shape))
        self.write(data.tobytes())

    def get_slices(self, slices):
        return self.get_data()[slices]

    def set_slices(self, slices, value):
        data = self.get_data()
        data[slices] = value
        self.write(data.tobytes())


    ###########################################################
    # CREATION
    ###########################################################

    @classmethod
    def create(cls, pool, name, shape=None, dtype=None, fillvalue=None, data=None, slices=None):
        if data is None:
            if fillvalue is None:
                cdata = np.zeros(shape, dtype)
            else:
                cdata = np.full(shape, fillvalue, dtype=dtype)
        else:
            if shape is None or data.shape == shape:
                cdata = data
                shape = data.shape
                dtype = data.dtype
            elif all(ds <= cs for ds, cs in zip(data.shape, shape)):
                cdata = np.full(shape, fillvalue, dtype=dtype)
                cdata[slices] = data
            else:
                raise DataChunkException('Data shape `{}` does not match chunk shape `{}`'
                                         .format(data.shape, shape))

        pool.write_full(name, cdata.tobytes())
        pool.set_xattr(name, 'shape', shape2str(shape))
        pool.set_xattr(name, 'dtype', dtype2str(dtype))
        return cls(pool, name)

    ###########################################################
    # BINDINGS to lower-level pool object
    ###########################################################

    def write(self, data):
        self.pool.write_full(self.name, data)

    def read(self, length=None, offset=0):
        if length is None:
            length = self.size * np.dtype(self.dtype).itemsize  # number of bytes
        return self.pool.read(self.name, length=length, offset=offset)