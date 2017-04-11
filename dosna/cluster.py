

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
    __random_pool_prefix__ = 'dosna_random_'
    __test_pool_prefix__ = 'test_dosna_'

    def __init__(self, conffile='ceph.conf', logger=logging, timeout=5):
        self.__cluster = None # Prevent exception tests from failling
        self.__cluster = rados.Rados(conffile=conffile)
        self.__connected = False
        self.__logger = logger
        self.__timeout = timeout

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
        self.__logger.info('Connected to cluster')
        self.__cluster.connect(timeout=self.__timeout)
        self.__connected = True
        return self

    def disconnect(self):
        self.__logger.info('Disconnected from cluster')
        self.__cluster.shutdown()
        self.__connected = False

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
        return self.__cluster

    @property
    def connected(self):
        return self.cluster is not None and self.__connected

    ###########################################################
    # POOLS
    ###########################################################

    def pools(self):
        return [self.get_pool(name) for name in self.cluster.list_pools()]

    list_pools = pools

    def create_pool(self, pool_name=None, open_mode='a',
                    auid=None, crush_rule=None, test=False):
        if pool_name is None:
            pool_prefix = "dosna_random_" if not test else 'test_dosna_'
            pool_name = pool_prefix + str(time.time())
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

    def __init__(self, name, open_mode='a', cluster=None, auto_open=True):
        self.name = name
        self.__is_open = False
        self.__ioctx = None
        self.__cluster = None
        self.__open_mode = self._parse_open_mode(open_mode)

        if cluster is None and not Cluster.instance().connected:
            raise ClusterException('Cluster object has not been initialized')
        elif cluster is None:
            self.__cluster = Cluster.instance()
        else:
            self.__cluster = cluster

        if not self.__cluster.has_pool(self.name):
            if not self.can_create:
                raise ClusterException('Error creating Pool {} with incorrect permissions {}'.format(self.name, open_mode))
            self.__cluster.cluster.create_pool(self.name)
        elif self.fail_exist:
            raise ClusterException('Error creating Pool {}, already exists'.format(self.name))
        elif self.truncate:
            self.__cluster.delete_pool(self.name)
            self.__cluster.create_pool(self.name)

        if auto_open:
            self.open()

    def __del__(self):
        if self.is_open:
            self.close()

    def delete(self):
        if self.is_open:
            self.close()
        self.__cluster.delete_pool(self.name)

    def __delitem__(self, key):
        self[key].delete()

    def _parse_open_mode(self, open_mode):
        return dict(
            read_only=(open_mode == 'r'),
            can_create=(open_mode in ['w', 'a', 'x', 'w-']),
            fail_exist=(open_mode in ['x', 'w-']),
            truncate=(open_mode == 'w')
        )

    @property
    def read_only(self):
        return self.__open_mode['read_only']

    @property
    def can_create(self):
        return self.__open_mode['can_create']

    @property
    def fail_exist(self):
        return self.__open_mode['fail_exist']

    @property
    def truncate(self):
        return self.__open_mode['truncate']

    ###########################################################
    # OPEN/CLOSE POOL
    ###########################################################

    @property
    def is_open(self):
        return self.__is_open

    def open(self):
        if self.is_open:
            raise ClusterException('Pool {} is already open'.format(self.name))
        self.__ioctx = self.__cluster.cluster.open_ioctx(self.name)
        self.__is_open = True

    def close(self):
        self.__ioctx.close()

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
            return self.__ioctx.__getattribute__(attr)
        raise ClusterException('Accessing low-level API of a non-open Pool {}'.format(self.name))

    ##########################################################
    # DATASET MANAGEMENT
    ###########################################################

    def __getitem__(self, name):
        if self.has_dataset(name):
            return Dataset(self, name)
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

    def __init__(self, name, open_mode='a'):
        super(File, self).__init__(name, open_mode=open_mode)

    def flush(self):
        pass