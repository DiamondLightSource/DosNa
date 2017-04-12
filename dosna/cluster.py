

import time
import rados
import logging
import os.path as op

from .dataset import Dataset, DatasetException


__default_conffile = op.realpath(op.join(op.dirname(__file__), '..', 'ceph.conf'))


def connect(conffile=__default_conffile, timeout=5):
    Cluster.instance(conffile=conffile, timeout=timeout).connect()


def disconnect():
    if Cluster.instance().connected:
        Cluster.instance().disconnect()


class ClusterException(Exception):
    pass


class Cluster(object):

    default_instance__ = None

    def __init__(self, njobs=1, conffile='ceph.conf', logger=logging, timeout=5):
        self._cluster = None  # Prevent exception tests from failling
        self._cluster = rados.Rados(conffile=conffile)
        self._connected = False
        self._logger = logger
        self._timeout = timeout
        self._njobs = njobs

    def __del__(self):
        if self.connected:
            self.disconnect()

    ###########################################################
    # CONNECTION
    ###########################################################

    @staticmethod
    def instance(*args, **kwargs):
        if Cluster.default_instance__ is None:
            Cluster.default_instance__ = Cluster(*args, **kwargs)
        return Cluster.default_instance__

    ###########################################################
    # CONNECTION
    ###########################################################

    def connect(self):
        self._logger.info('Connected to cluster')
        self._cluster.connect(timeout=self._timeout)
        self._connected = True
        return self

    def disconnect(self):
        self._logger.info('Disconnected from cluster')
        self._cluster.shutdown()
        self._connected = False

    ###########################################################
    # SAFE CONTEXT
    ###########################################################

    def __enter__(self):
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connected:
            self.disconnect()

    ###########################################################
    # PROEPRTIES
    ###########################################################

    @property
    def cluster(self):
        return self._cluster

    @property
    def connected(self):
        return self.cluster is not None and self._connected

    @property
    def njobs(self):
        return self._njobs

    ###########################################################
    # POOLS
    ###########################################################

    def pools(self):
        return [self.get_pool(name) for name in self.cluster.list_pools()]

    list_pools = pools

    def create_pool(self, pool_name=None, open_mode='a',
                    auid=None, crush_rule=None, test=False):
        if pool_name is None:
            pool_name = Pool.random_name(test=test)
        if self.has_pool(pool_name):
            raise ClusterException('Pool {} already exists'.format(pool_name))
        self.cluster.create_pool(pool_name, auid=auid, crush_rule=crush_rule)
        return self.get_pool(pool_name, open_mode=open_mode)

    def delete_pool(self, pool_name):
        if self.has_pool(pool_name):
            self.cluster.delete_pool(pool_name)
            return True
        return False

    def has_pool(self, pool_name):
        return self.cluster.pool_exists(pool_name)

    def get_pool(self, pool_name, open_mode='a'):
        if self.has_pool(pool_name):
            return Pool(pool_name, open_mode=open_mode, cluster=self)
        raise ClusterException('Pool {} doesnt exist'.format(pool_name))

    def __getitem__(self, pool_name):
        return self.get_pool(pool_name)


class Pool(object):

    __random_pool_prefix__ = 'dosna_random_'
    __test_pool_prefix__ = 'test_dosna_'

    def __init__(self, name=None, open_mode='a', cluster=None, auto_open=True, njobs=None, test=False):
        self.name = name or Pool.random_name(test=test)
        self._is_open = False
        self._ioctx = None
        self._cluster = None
        self._open_mode = self._parse_open_mode(open_mode)

        if cluster is None and not Cluster.instance().connected:
            raise ClusterException('Cluster object has not been initialized')
        elif cluster is None:
            self._cluster = Cluster.instance()
        else:
            self._cluster = cluster

        self._njobs = self._cluster.njobs if njobs is None else njobs

        if not self._cluster.has_pool(self.name):
            if not self.can_create:
                raise ClusterException('Error creating Pool {} with incorrect permissions {}'.format(self.name, open_mode))
            self._cluster.cluster.create_pool(self.name)
        elif self.fail_exist:
            raise ClusterException('Error creating Pool {}, already exists'.format(self.name))
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

    def _parse_open_mode(self, open_mode):
        return dict(
            read_only=(open_mode == 'r'),
            can_create=(open_mode in ['w', 'a', 'x', 'w-']),
            fail_exist=(open_mode in ['x', 'w-']),
            truncate=(open_mode == 'w')
        )

    @staticmethod
    def random_name(test=False):
        if test:
            prefix = Pool.__test_pool_prefix__
        else:
            prefix = Pool.__random_pool_prefix__
        return prefix + str(time.time())

    @property
    def read_only(self):
        return self._open_mode['read_only']

    @property
    def can_create(self):
        return self._open_mode['can_create']

    @property
    def fail_exist(self):
        return self._open_mode['fail_exist']

    @property
    def truncate(self):
        return self._open_mode['truncate']

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
            raise ClusterException('Pool {} is already open'.format(self.name))
        self._ioctx = self._cluster.cluster.open_ioctx(self.name)
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
        raise ClusterException('Accessing low-level API of a non-open Pool {}'.format(self.name))

    ##########################################################
    # DATASET MANAGEMENT
    ###########################################################

    def __getitem__(self, name):
        if self.has_dataset(name):
            return Dataset(self, name, njobs=self.njobs)
        raise DatasetException('No dataset {} found in pool {}'.format(name, self.name))

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

    def create_dataset(self, name, shape=None, dtype=None, fillvalue=-1,
                       data=None, chunks=None, read_only=False):
        if self.read_only:
            raise ClusterException('Error creating dataset, Pool {} is read-only'.format(self.name))
        return Dataset.create(self, name, shape=shape, dtype=dtype,
                              fillvalue=fillvalue, chunks=chunks,
                              data=data, read_only=read_only)

    def delete_dataset(self, name):
        if self.read_only:
            raise ClusterException('Error creating dataset, Pool {} is read-only'.format(self.name))
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

    def __init__(self, name=None, open_mode='a', test=False):
        super(File, self).__init__(name=name, open_mode=open_mode, test=test)

    def flush(self):
        pass