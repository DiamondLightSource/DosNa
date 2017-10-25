

import os
import shutil

import logging as log

import h5py as h5
import numpy as np

from .. import Backend
from ..base import BaseCluster, BasePool, BaseDataset, BaseDataChunk
from ..utils import DirectoryTreeMixin
from ..utils import dtype2str

_DATASET_METADATA_FILENAME = 'dataset.h5'
_POOL_SIGNATURE_FILENAME = '.dosna'


def _validate_path(path):
    if len(os.path.splitext(path)[1]) > 0:
        raise Exception('`%s` is not a valid cluster/pool path' % path)


class H5Cluster(BaseCluster, DirectoryTreeMixin):
    """
    A HDF5 Cluster represents the local filesystem.
    """

    def __init__(self, name, directory='/tmp', *args, **kwargs):
        super(H5Cluster, self).__init__(name, *args, **kwargs)
        self.directory = os.path.realpath(directory)
        _validate_path(self.path)

    def _get_pool_signature_path(self, name):
        return os.path.join(self.relpath(name), _POOL_SIGNATURE_FILENAME)

    def connect(self):
        super(H5Cluster, self).connect()
        log.debug('Starting HDF5 Cluster at `%s`' % self.path)

    def disconnect(self):
        super(H5Cluster, self).disconnect()
        log.debug('Stopping HDF5 Cluster at `%s`' % self.path)

    def create_pool(self, name, open_mode='a'):
        path = self.relpath(name)
        log.debug('Creating pool `%s`' % path)

        if not os.path.exists(path):
            os.makedirs(path)
        flag_path = self._get_pool_signature_path(name)
        with open(flag_path, 'w'):
            os.utime(flag_path, None)
        return H5Pool(self, name, open_mode=open_mode)

    def get_pool(self, name, open_mode='a'):
        if self.has_pool(name):
            return H5Pool(self, name, open_mode=open_mode)
        path = self.relpath(name)
        if os.path.exists(path):
            raise Exception('Path `%s` is not a pool' % path)
        raise Exception('Pool `%s` does not exist' % path)

    def has_pool(self, name):
        return os.path.isdir(self.relpath(name)) \
               and os.path.isfile(self._get_pool_signature_path(name))

    def del_pool(self, name):
        path = self.relpath(name)
        if self.has_pool(name):
            log.debug('Removing pool at `%s`' % path)
            shutil.rmtree(path)
        if os.path.exists(path):
            raise Exception('Path `%s` is not a valid pool' % path)


class H5Pool(BasePool, DirectoryTreeMixin):
    """
    An HDF5 Pool represents a subdirectory in the local filesystem with a `.dosna` file.
    """

    def __init__(self, *args, **kwargs):
        super(H5Pool, self).__init__(*args, **kwargs)
        self.parent = self.cluster
        _validate_path(self.path)

    def _get_dataset_metadata_path(self, name):
        return os.path.join(self.relpath(name), _DATASET_METADATA_FILENAME)

    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunks=None):

        if not ((shape is not None and dtype is not None) or data is not None):
            raise Exception('Provide `shape` and `dtype` or `data`')
        if self.has_dataset(name):
            raise Exception('Dataset at `%s` already exists' % self.relpath(name))

        if data is not None:
            shape = data.shape
            dtype = data.dtype

        if chunks is None:
            chunk_size = shape
        else:
            chunk_size = chunks
        chunks_needed = (np.ceil(np.asarray(shape, float) / chunk_size)).astype(int)

        path = self.relpath(name)
        os.mkdir(path)
        with h5.File(self._get_dataset_metadata_path(name), 'w') as f:
            f.attrs['shape'] = shape
            f.attrs['dtype'] = dtype2str(dtype)
            f.attrs['fillvalue'] = np.dtype(dtype).type(fillvalue)
            f.attrs['chunks'] = np.asarray(chunks_needed, dtype=int)
            f.attrs['chunk_size'] = np.asarray(chunk_size, dtype=int)

        log.debug('Creating dataset at `%s`' % path)

        dataset = H5Dataset(self, name, shape, dtype, fillvalue,
                            chunks_needed, chunk_size)

        return dataset

    def get_dataset(self, name):
        if not self.has_dataset(name):
            raise Exception('Dataset at `%s` does not exist' % self.relpath(name))

        with h5.File(self._get_dataset_metadata_path(name), 'r') as f:
            shape = tuple(f.attrs['shape'])
            dtype = f.attrs['dtype']
            fillvalue = f.attrs['fillvalue']
            chunks_needed = f.attrs['chunks']
            chunk_size = f.attrs['chunk_size']

        return H5Dataset(self, name, shape, dtype, fillvalue, chunks_needed, chunk_size)

    def has_dataset(self, name):
        return os.path.isdir(self.relpath(name)) \
               and os.path.isfile(self._get_dataset_metadata_path(name))

    def del_dataset(self, name):
        path = self.relpath(name)
        if not self.has_dataset(name):
            raise Exception('Dataset at `%s` does not exist' % path)
        log.debug('Removing Dataset at `%s`' % path)
        shutil.rmtree(path)


class H5Dataset(BaseDataset, DirectoryTreeMixin):

    def __init__(self, *args, **kwargs):
        super(H5Dataset, self).__init__(*args, **kwargs)
        self.parent = self.pool
        self._subchunks = kwargs.pop('subchunks', None)

    def _idx2name(self, idx):
        if not all([type(i) == int for i in idx]) or len(idx) != self.ndim:
            raise Exception('Invalid chunk idx')
        return 'chunk_%s.h5' % '_'.join(map(str, idx))

    def create_chunk(self, idx, data=None, slices=None):
        if self.has_chunk(idx):
            raise Exception('DataChunk `{}` already exists'.format(idx))

        chunk_name = self._idx2name(idx)
        with h5.File(self.relpath(chunk_name), 'w') as f:
            f.create_dataset('data', shape=self.chunk_size, dtype=self.dtype,
                             fillvalue=self.fillvalue, chunks=self._subchunks)
            if data is not None:
                slices = slices or slice(None)
                f['data'][slices] = data

        return H5DataChunk(self, idx, chunk_name, self.chunk_size, self.dtype,
                           self.fillvalue)

    def get_chunk(self, idx):
        if self.has_chunk(idx):
            chunk_name = self._idx2name(idx)
            with h5.File(self.relpath(chunk_name), 'r') as f:
                shape = f['data'].shape
                dtype = f['data'].dtype
            return H5DataChunk(self, idx, chunk_name, shape, dtype, self.fillvalue)
        return self.create_chunk(idx)

    def has_chunk(self, idx):
        return os.path.isfile(self.relpath(self._idx2name(idx)))

    def del_chunk(self, idx):
        if self.has_chunk(idx):
            os.remove(self.relpath(self._idx2name(idx)))


class H5DataChunk(BaseDataChunk, DirectoryTreeMixin):

    def __init__(self, *args, **kwargs):
        super(H5DataChunk, self).__init__(*args, **kwargs)
        self.parent = self.dataset

    def get_data(self, slices=None):
        if slices is None:
            slices = slice(None)

        with h5.File(self.path, 'r') as f:
            data = f['data'][slices]

        return data

    def set_data(self, values, slices=None):
        if slices is None:
            slices = slice(None)

        with h5.File(self.path, 'a') as f:
            f['data'][slices] = values


__backend__ = Backend('hdf5', H5Cluster, H5Pool, H5Dataset, H5DataChunk)
