

import rados
import logging
import os.path as op

import pool as dnPool  # full import to avoid cyclic-imports


__default_conffile = op.realpath(op.join(op.dirname(__file__), '..', 'ceph.conf'))


def connect(**kwargs):
    kwargs.setdefault('conffile', __default_conffile)
    C = Cluster.instance(**kwargs)
    if C.connected:
        raise ClusterException('Default cluster instance is already connected.')
    else:
        C.connect()


def disconnect():
    if Cluster.instance().connected:
        Cluster.instance().disconnect()
    else:
        logging.warning('Nothing to disconnect. Default cluster instance is not connected.')


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
    def connected(self):
        return self._cluster is not None and self._connected

    @property
    def njobs(self):
        return self._njobs

    ###########################################################
    # POOLS
    ###########################################################

    def pools(self):
        return [self.get_pool(name) for name in self._cluster.list_pools()]

    list_pools = pools

    def create_pool(self, pool_name=None, auid=None, crush_rule=None, test=False, **kwargs):
        pool_name = pool_name or dnPool.Pool.random_name(test=test)
        if self.has_pool(pool_name):
            raise ClusterException('Pool {} already exists'.format(pool_name))
        self._cluster.create_pool(pool_name, auid=auid, crush_rule=crush_rule)
        return self.get_pool(pool_name, **kwargs)

    def delete_pool(self, pool_name):
        if self.has_pool(pool_name):
            self._cluster.delete_pool(pool_name)
            return True
        return False

    def has_pool(self, pool_name):
        return self._cluster.pool_exists(pool_name)

    def get_pool(self, pool_name, **kwargs):
        if self.has_pool(pool_name):
            return dnPool.Pool(name=pool_name, cluster=self, **kwargs)
        raise ClusterException('Pool {} doesnt exist'.format(pool_name))

    def open_ioctx(self, name):
        return self._cluster.open_ioctx(name)

    def __getitem__(self, pool_name):
        return self.get_pool(pool_name)


# Quick bind to get cluster instance
instance = Cluster.instance
