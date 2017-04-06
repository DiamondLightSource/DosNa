

import rados
import logging

from .dataset import Dataset, DatasetException


class ClusterException(Exception):
    pass


class Pool(object):

    def __init__(self, cluster, name):
        self.name = name
        self.__cluster = cluster
        self.__ioctx = cluster.open_ioctx(name)

    def __getattr__(self, attr):
        return self.__ioctx.__getattribute__(attr)

    def __del__(self):
        self.__ioctx.close()

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
                       data=None, chunks=None):
        return Dataset.create(self, name, shape=shape, dtype=dtype,
                              fillvalue=fillvalue, chunks=chunks,
                              data=data)

    def delete_dataset(self, name):
        self[name].delete()

    def has_dataset(self, name):
        try:
            valid = (self.stat(name)[0] == len(Dataset.__signature__) \
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

    def delete(self):
        self.__cluster.delete_pool(self.name)


class Cluster(object):

    def __init__(self, conffile='ceph.conf', logger=logging):
        self.__cluster = None # Prevent exception tests from failling
        self.__cluster = rados.Rados(conffile=conffile)
        self.__connected = False
        self.__logger = logger

    def __del__(self):
        if self.connected:
            self.disconnect()

    ###########################################################
    # CONNECTION
    ###########################################################

    def connect(self):
        self.__logger.info('Connected to cluster')
        self.__cluster.connect()
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
        return self.cluster.list_pools()

    def create_pool(self, pool_name, auid=None, crush_rule=None):
        if self.has_pool(pool_name):
            raise ClusterException('Pool {} already exists'.format(pool_name))
        self.cluster.create_pool(pool_name, auid=auid, crush_rule=crush_rule)
        return self.get_pool(pool_name)

    def delete_pool(self, pool_name):
        if self.has_pool(pool_name):
            self.cluster.delete_pool(pool_name)
            return True
        return False

    def has_pool(self, pool_name):
        return self.cluster.pool_exists(pool_name)

    def get_pool(self, pool_name):
        if self.has_pool(pool_name):
            return Pool(self.cluster, pool_name)
        raise ClusterException('Pool {} doesnt exist'.format(pool_name))

    def __getitem__(self, pool_name):
        return self.get_pool(pool_name)
