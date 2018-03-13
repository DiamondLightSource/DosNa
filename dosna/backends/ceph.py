#!/usr/bin/env python
"""Backend ceph uses a ceph cluster to store the dataset and chunks data"""

import logging

import numpy as np

import rados
from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset, ConnectionError,
                                 DatasetNotFoundError)
from dosna.util import dtype2str, shape2str, str2shape
from dosna.util.data import slices2shape

_SIGNATURE = "DosNa Dataset"

log = logging.getLogger(__name__)


class CephConnection(BackendConnection):
    """
    A Ceph Cluster that wraps LibRados.Cluster
    """

    def __init__(self, name, conffile='ceph.conf', timeout=5,
                 client_id=None, *args, **kwargs):
        super(CephConnection, self).__init__(name, *args, **kwargs)

        rados_options = {
            "conffile": conffile
        }
        if client_id is not None:
            client_name = "client.{}".format(client_id)
            rados_options["name"] = client_name

        self._cluster = rados.Rados(**rados_options)
        self._timeout = timeout
        self._ioctx = None

    def connect(self):
        if self.connected:
            raise ConnectionError(
                'Connection {} is already open'.format(self.name))
        self._cluster.connect(timeout=self._timeout)
        self._ioctx = self._cluster.open_ioctx(self.name)
        super(CephConnection, self).connect()

    def disconnect(self):
        if self.connected:
            self.ioctx.close()
            self._cluster.shutdown()
            super(CephConnection, self).disconnect()

    @property
    def ioctx(self):
        return self._ioctx

    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunk_size=None):
        if not ((shape is not None and dtype is not None) or data is not None):
            raise Exception('Provide `shape` and `dtype` or `data`')
        if self.has_dataset(name):
            raise Exception('Dataset `%s` already exists' % name)

        if data is not None:
            shape = data.shape
            dtype = data.dtype

        if chunk_size is None:
            chunk_size = shape

        chunk_grid = (np.ceil(np.asarray(shape, float) / chunk_size))\
            .astype(int)

        log.debug('creating dataset %s with shape:%s chunk_size:%s '
                  'chunk_grid:%s', name, shape, chunk_size, chunk_grid)

        self.ioctx.write(name, _SIGNATURE)
        self.ioctx.set_xattr(name, 'shape', shape2str(shape))
        self.ioctx.set_xattr(name, 'dtype', dtype2str(dtype))
        self.ioctx.set_xattr(name, 'fillvalue', repr(fillvalue))
        self.ioctx.set_xattr(name, 'chunk_grid', shape2str(chunk_grid))
        self.ioctx.set_xattr(name, 'chunk_size', shape2str(chunk_size))

        dataset = CephDataset(self, name, shape, dtype, fillvalue,
                              chunk_grid, chunk_size)

        return dataset

    def get_dataset(self, name):
        if not self.has_dataset(name):
            raise DatasetNotFoundError('Dataset `%s` does not exist' % name)
        shape = str2shape(self.ioctx.get_xattr(name, 'shape'))
        dtype = self.ioctx.get_xattr(name, 'dtype')
        fillvalue = int(self.ioctx.get_xattr(name, 'fillvalue'))
        chunk_grid = str2shape(self.ioctx.get_xattr(name, 'chunk_grid'))
        chunk_size = str2shape(self.ioctx.get_xattr(name, 'chunk_size'))
        dataset = CephDataset(self, name, shape, dtype, fillvalue,
                              chunk_grid, chunk_size)
        return dataset

    def has_dataset(self, name):
        try:
            valid = self.ioctx.stat(name)[0] == len(_SIGNATURE) and \
                self.ioctx.read(name) == _SIGNATURE
        except rados.ObjectNotFound:
            return False
        return valid

    def del_dataset(self, name):
        log.debug("Removing dataset %s", name)
        if self.has_dataset(name):
            self.ioctx.remove_object(name)
        else:
            raise DatasetNotFoundError(
                'Dataset `{}` does not exist'.format(name))


class CephDataset(BackendDataset):
    """
    CephDataset wraps an instance of Rados.Object
    """

    @property
    def ioctx(self):
        return self.connection.ioctx

    def _idx2name(self, idx):
        return '{}/{}'.format(self.name, '.'.join(map(str, idx)))

    def create_chunk(self, idx, data=None, slices=None):
        if self.has_chunk(idx):
            raise Exception('DataChunk `{}{}` already exists'.format(self.name,
                                                                     idx))
        name = self._idx2name(idx)
        dtype = self.dtype
        shape = self.chunk_size
        fillvalue = self.fillvalue
        datachunk = CephDataChunk(self, idx, name, shape, dtype, fillvalue)
        if data is None:
            data = np.full(shape, fillvalue, dtype)
        datachunk.set_data(data, slices)
        return datachunk

    def get_chunk(self, idx):
        if self.has_chunk(idx):
            name = self._idx2name(idx)
            dtype = self.dtype
            shape = self.chunk_size
            fillvalue = self.fillvalue
            return CephDataChunk(self, idx, name, shape, dtype, fillvalue)
        return self.create_chunk(idx)

    def has_chunk(self, idx):
        name = self._idx2name(idx)
        try:
            self.ioctx.stat(name)
        except rados.ObjectNotFound:
            return False
        return True

    def del_chunk(self, idx):
        if self.has_chunk(idx):
            self.ioctx.remove_object(self._idx2name(idx))


class CephDataChunk(BackendDataChunk):

    @property
    def ioctx(self):
        return self.dataset.ioctx

    def get_data(self, slices=None):
        if slices is None:
            slices = slice(None)
        data = np.fromstring(self.read(), dtype=self.dtype, count=self.size)
        data.shape = self.shape
        return data[slices]

    def set_data(self, values, slices=None):
        if slices is None or slices2shape(slices) == self.shape:
            self.write_full(values.tobytes())
        else:
            cdata = self.get_data()
            cdata[slices] = values
            self.write_full(cdata.tobytes())

    def write_full(self, data):
        self.ioctx.write_full(self.name, data)

    def read(self, length=None, offset=0):
        if length is None:
            length = self.byte_count
        return self.ioctx.read(self.name, length=length, offset=offset)


_backend = Backend('ceph', CephConnection, CephDataset, CephDataChunk)
