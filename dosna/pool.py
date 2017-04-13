

import rados
import time

import cluster as dnCluster  # full import to avoid cyclic-imports
from .dataset import Dataset


class PoolException(Exception):
    pass


class Pool(object):

    __random_pool_prefix__ = 'dosna_random_'
    __test_pool_prefix__ = 'test_dosna_'

    def __init__(self, name=None, open_mode='a', cluster=None, auto_open=True,
                 njobs=None, test=False):
        self.name = name or Pool.random_name(test=test)
        self._is_open = False
        self._ioctx = None
        self._cluster = None
        self._open_mode = open_mode

        self._cluster = cluster or dnCluster.instance()
        if not self._cluster.connected:
            raise PoolException('Cluster object is not connected')

        self._njobs = njobs or self._cluster.njobs

        if not self._cluster.has_pool(self.name):
            if not self.can_create:
                raise PoolException('Error creating Pool `{}` with incorrect permissions `{}`'
                                    .format(self.name, open_mode))
            self._cluster.create_pool(self.name)
        elif self.fail_exist:
            raise PoolException('Error creating Pool `{}`, already exists'.format(self.name))
        elif self.truncate:
            self._cluster.delete_pool(self.name)
            self._cluster.create_pool(self.name)

        if auto_open:
            self.open()

    def __del__(self):
        if self.is_open:
            self.close()

    def delete(self):
        if self.is_open:
            self.close()
        self._cluster.delete_pool(self.name)

    def __delitem__(self, key):
        self[key].delete()

    @staticmethod
    def random_name(test=False):
        if test:
            prefix = Pool.__test_pool_prefix__
        else:
            prefix = Pool.__random_pool_prefix__
        return prefix + str(time.time())

    @property
    def read_only(self):
        return self._open_mode == 'r'

    @property
    def can_create(self):
        return self._open_mode in ['w', 'a', 'x', 'w-']

    @property
    def fail_exist(self):
        return self._open_mode in ['x', 'w-']

    @property
    def truncate(self):
        return self._open_mode == 'w'

    @property
    def njobs(self):
        return self._njobs

    ###########################################################
    # OPEN/CLOSE POOL
    ###########################################################

    @property
    def is_open(self):
        return self._is_open

    def open(self):
        if self.is_open:
            raise PoolException('Pool {} is already open'.format(self.name))
        self._ioctx = self._cluster.open_ioctx(self.name)
        self._is_open = True

    def close(self):
        self._ioctx.close()

    ###########################################################
    # SAFE CONTEXT
    ###########################################################

    def __enter__(self):
        if not self.is_open:
            self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_open:
            self.close()

    ##########################################################
    # LOW-LEVEL interface to librados
    ###########################################################

    def __getattr__(self, attr):
        if self.is_open:
            return self._ioctx.__getattribute__(attr)
        raise PoolException('Accessing low-level API of a non-open Pool `{}`'.format(self.name))

    ##########################################################
    # DATASET MANAGEMENT
    ###########################################################

    def __getitem__(self, name):
        if self.has_dataset(name):
            return Dataset(self, name, njobs=self.njobs)
        raise PoolException('No dataset `{}` found in pool `{}`'.format(name, self.name))

    def list_datasets(self):
        datasets = []
        for obj in self.list_objects():
            if self.stat(obj.key)[0] == len(Dataset.__signature__) \
                    and self.read(obj.key) == Dataset.__signature__:
                datasets.append(obj.key)
        return datasets

    def dataset_count(self):
        return len(self.list_datasets())

    def object_count(self):
        return len(list(self.list_objects()))

    def create_dataset(self, name, shape=None, dtype=None, **kwargs):
        if self.read_only:
            raise PoolException('Error creating dataset, Pool `{}` is read-only'.format(self.name))
        return Dataset.create(self, name, shape=shape, dtype=dtype, **kwargs)

    def zeros(self, name, shape=None, dtype=None, **kwargs):
        return self.create_dataset(name, shape=shape, dtype=dtype,
                                   fillvalue=0, data=None, **kwargs)

    def ones(self, name, shape=None, dtype=None, **kwargs):
        return self.create_dataset(name, shape=shape, dtype=dtype,
                                   fillvalue=1, data=None, **kwargs)

    def create_like(self, name, data, **kwargs):
        if self.read_only:
            raise PoolException('Error creating dataset, Pool `{}` is read-only'.format(self.name))
        return Dataset.create_like(name, data, pool=self, **kwargs)

    def zeros_like(self, name, data, **kwargs):
        return self.create_like(name, data, fillvalue=0, **kwargs)

    def ones_like(self, name, data, **kwargs):
        return self.create_like(name, data, fillvalue=1, **kwargs)

    def delete_dataset(self, name):
        if self.read_only:
            raise PoolException('Error deleting dataset, Pool `{}` is read-only'.format(self.name))
        self[name].delete()

    def has_dataset(self, name):
        try:
            valid = (self.stat(name)[0] == len(Dataset.__signature__)
                     and self.read(name) == Dataset.__signature__)
        except rados.ObjectNotFound:
            return False
        return valid

    def has_chunk(self, name):
        try:
            self.stat(name)
        except rados.ObjectNotFound:
            return False
        return True


class File(Pool):
    """
    Compatibility with h5py: see examples/basic_h5py_compat.py
    """

    def __init__(self, name=None, open_mode='a', **kwargs):
        super(File, self).__init__(name=name, open_mode=open_mode, **kwargs)

    def flush(self):
        pass