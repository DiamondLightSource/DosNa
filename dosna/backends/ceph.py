#!/usr/bin/env python
"""Backend ceph uses a ceph cluster to store the dataset and chunks data"""

import logging

import numpy as np

import rados
from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset)
from dosna.utils import dtype2str, shape2str, str2shape

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
            raise Exception('Connection {} is already open'.format(self.name))
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
                       data=None, chunks=None):

        if not ((shape is not None and dtype is not None) or data is not None):
            raise Exception('Provide `shape` and `dtype` or `data`')
        if self.has_dataset(name):
            raise Exception('Dataset `%s` already exists' % name)

        if data is not None:
            shape = data.shape
            dtype = data.dtype

        if chunks is None:
            csize = shape
        else:
            csize = chunks
        chunks_needed = (np.ceil(np.asarray(shape, float) / csize)).astype(int)

        self.ioctx.write(name, _SIGNATURE)
        self.ioctx.set_xattr(name, 'shape', shape2str(shape))
        self.ioctx.set_xattr(name, 'dtype', dtype2str(dtype))
        self.ioctx.set_xattr(name, 'fillvalue', repr(fillvalue))
        self.ioctx.set_xattr(name, 'chunks', shape2str(chunks_needed))
        self.ioctx.set_xattr(name, 'chunk_size', shape2str(csize))

        dataset = CephDataset(self, name, shape, dtype, fillvalue,
                              chunks_needed, csize)

        return dataset

    def get_dataset(self, name):
        if not self.has_dataset(name):
            raise Exception('Dataset `%s` does not exist' % name)
        shape = str2shape(self.ioctx.get_xattr(name, 'shape'))
        dtype = self.ioctx.get_xattr(name, 'dtype')
        fillvalue = int(self.ioctx.get_xattr(name, 'fillvalue'))
        chunks = str2shape(self.ioctx.get_xattr(name, 'chunks'))
        chunk_size = str2shape(self.ioctx.get_xattr(name, 'chunk_size'))
        dataset = CephDataset(self, name, shape, dtype, fillvalue,
                              chunks, chunk_size)
        return dataset

    def has_dataset(self, name):
        try:
            valid = self.ioctx.stat(name)[0] == len(_SIGNATURE) and \
                self.ioctx.read(name) == _SIGNATURE
        except rados.ObjectNotFound:
            return False
        return valid

    def del_dataset(self, name):
        if self.has_dataset(name):
            self.ioctx.remove_object(name)
        else:
            raise Exception('Dataset `{}` does not exist'.format(name))


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
            raise Exception('DataChunk `{}` already exists'.format(idx))
        name = self._idx2name(idx)
        dtype = self.dtype
        shape = self.chunk_size
        fillvalue = self.fillvalue
        cdata = np.full(shape, fillvalue, dtype)
        if data is not None:
            cdata[slices] = data
        self.ioctx.write_full(name, cdata.tobytes())
        return CephDataChunk(self, idx, name, shape, dtype, fillvalue)

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
        data_count = np.prod(self.shape)
        data = np.fromstring(self.read(), dtype=self.dtype, count=data_count)
        data.shape = self.shape
        return data[slices]

    def set_data(self, values, slices=None):
        if slices is None:
            slices = slice(None)
        cdata = self.get_data()
        cdata[slices] = values
        self.write_full(cdata.tobytes())

    def write_full(self, data):
        self.ioctx.write_full(self.name, data)

    def read(self, length=None, offset=0):
        if length is None:
            length = self.size * np.dtype(self.dtype).itemsize
        return self.ioctx.read(self.name, length=length, offset=offset)


_backend = Backend('ceph', CephConnection, CephDataset, CephDataChunk)
