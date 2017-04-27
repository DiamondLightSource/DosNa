

import numpy as np
import joblib

from .utils import str2shape


class ParallelMixin(object):

    _njobs = 1

    @property
    def njobs(self):
        return self._njobs

    @njobs.setter
    def njobs(self, n):
        self._njobs = n if n > 0 else joblib.cpu_count()


class BaseData(object):

    def __init__(self, pool, name, read_only):
        self._pool = pool
        self.name = name
        self.read_only = read_only

        self._shape = str2shape(self._pool.get_xattr(self.name, 'shape'))
        self._ndim = len(self._shape)
        self._size = np.prod(self._shape)
        self._dtype = self._pool.get_xattr(self.name, 'dtype')
        self._itemsize = np.dtype(self._dtype).itemsize

    ###########################################################
    # PROPERTIES
    ###########################################################

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._ndim

    @property
    def size(self):
        return self._size

    @property
    def dtype(self):
        return self._dtype

    @property
    def itemsize(self):
        return self._itemsize

    @property
    def pool(self):
        return self._pool

    ###########################################################
    # BINDINGS to lower-level pool
    ###########################################################

    def get_xattrs(self):
        return self._pool.get_xattrs(self.name)

    def get_xattr(self, name):
        return self._pool.get_xattr(self.name, name)

    def set_xattr(self, name, value):
        return self._pool.set_xattr(self.name, name, value)

    def rm_xattr(self, name):
        return self._pool.rm_xattr(self.name, name)

    def stat(self):
        return self._pool.stat(self.name)

    def delete(self):
        return self._pool.remove_object(self.name)