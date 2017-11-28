#!/usr/bin/env python
"""Backend hdf5 uses hdf files to store the dataset and chunks data"""

import logging
import os
import shutil

import h5py as h5
import numpy as np

from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset, DatasetNotFoundError)
from dosna.util import DirectoryTreeMixin, dtype2str

_DATASET_METADATA_FILENAME = 'dataset.h5'
log = logging.getLogger(__name__)


def _validate_path(path):
    if os.path.splitext(path)[1]:
        raise Exception('`{}` is not a valid path'.format(path))


class H5Connection(BackendConnection, DirectoryTreeMixin):
    """
    A HDF5 Cluster represents the local filesystem.
    """

    def __init__(self, name, directory='.', *args, **kwargs):
        super(H5Connection, self).__init__(name, *args, **kwargs)
        self.directory = os.path.realpath(directory)
        _validate_path(self.path)

    def connect(self):
        super(H5Connection, self).connect()
        log.debug('Starting HDF5 Connection at `%s`', self.path)

    def disconnect(self):
        super(H5Connection, self).disconnect()
        log.debug('Stopping HDF5 Connection at `%s`', self.path)

    def _get_dataset_metadata_path(self, name):
        return os.path.join(self.relpath(name), _DATASET_METADATA_FILENAME)

    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunk_size=None):

        if not ((shape is not None and dtype is not None) or data is not None):
            raise Exception('Provide `shape` and `dtype` or `data`')
        if self.has_dataset(name):
            raise Exception(
                'Dataset at `{}` already exists'.format(self.relpath(name)))

        if data is not None:
            shape = data.shape
            dtype = data.dtype

        if chunk_size is None:
            chunk_size = shape

        chunk_grid = (np.ceil(np.asarray(shape, float) / chunk_size))\
            .astype(int)

        path = self.relpath(name)
        os.mkdir(path)
        with h5.File(self._get_dataset_metadata_path(name),
                     'w') as file_handle:
            file_handle.attrs['shape'] = shape
            file_handle.attrs['dtype'] = dtype2str(dtype)
            file_handle.attrs['fillvalue'] = np.dtype(dtype).type(fillvalue)
            file_handle.attrs['chunk_grid'] = np.asarray(chunk_grid, dtype=int)
            file_handle.attrs['chunk_size'] = np.asarray(chunk_size, dtype=int)

        log.debug('Creating dataset at `%s`', path)

        dataset = H5Dataset(self, name, shape, dtype, fillvalue,
                            chunk_grid, chunk_size)

        return dataset

    def get_dataset(self, name):
        if not self.has_dataset(name):
            raise DatasetNotFoundError(
                'Dataset at `%s` does not exist' % self.relpath(name))

        with h5.File(self._get_dataset_metadata_path(name),
                     'r') as file_handle:
            shape = tuple(file_handle.attrs['shape'])
            dtype = file_handle.attrs['dtype']
            fillvalue = file_handle.attrs['fillvalue']
            chunk_grid = file_handle.attrs['chunk_grid']
            chunk_size = file_handle.attrs['chunk_size']

        return H5Dataset(self, name, shape, dtype, fillvalue, chunk_grid,
                         chunk_size)

    def has_dataset(self, name):
        return os.path.isdir(self.relpath(name)) \
               and os.path.isfile(self._get_dataset_metadata_path(name))

    def del_dataset(self, name):
        path = self.relpath(name)
        if not self.has_dataset(name):
            raise DatasetNotFoundError(
                'Dataset at `{}` does not exist'.format(path))
        log.debug('Removing Dataset at `%s`', path)
        shutil.rmtree(path)


class H5Dataset(BackendDataset, DirectoryTreeMixin):

    def __init__(self, *args, **kwargs):
        super(H5Dataset, self).__init__(*args, **kwargs)
        self.parent = self.connection
        self._subchunks = kwargs.pop('subchunks', None)

    def _idx2name(self, idx):
        if not all([isinstance(i, int) for i in idx]) or len(idx) != self.ndim:
            raise Exception('Invalid chunk idx')
        return 'chunk_{}.h5'.format('_'.join(map(str, idx)))

    def create_chunk(self, idx, data=None, slices=None):
        if self.has_chunk(idx):
            raise Exception('DataChunk `{}` already exists'.format(idx))

        chunk_name = self._idx2name(idx)
        with h5.File(self.relpath(chunk_name), 'w') as file_handle:
            file_handle.create_dataset('data',
                                       shape=self.chunk_size,
                                       dtype=self.dtype,
                                       fillvalue=self.fillvalue,
                                       chunks=self._subchunks)
            if data is not None:
                slices = slices or slice(None)
                file_handle['data'][slices] = data

        return H5DataChunk(self, idx, chunk_name, self.chunk_size, self.dtype,
                           self.fillvalue)

    def get_chunk(self, idx):
        if self.has_chunk(idx):
            chunk_name = self._idx2name(idx)
            with h5.File(self.relpath(chunk_name), 'r') as file_handle:
                shape = file_handle['data'].shape
                dtype = file_handle['data'].dtype
            return H5DataChunk(self, idx, chunk_name, shape, dtype,
                               self.fillvalue)
        return self.create_chunk(idx)

    def has_chunk(self, idx):
        return os.path.isfile(self.relpath(self._idx2name(idx)))

    def del_chunk(self, idx):
        if self.has_chunk(idx):
            os.remove(self.relpath(self._idx2name(idx)))


class H5DataChunk(BackendDataChunk, DirectoryTreeMixin):

    def __init__(self, *args, **kwargs):
        super(H5DataChunk, self).__init__(*args, **kwargs)
        self.parent = self.dataset

    def get_data(self, slices=None):
        if slices is None:
            slices = slice(None)

        with h5.File(self.path, 'r') as file_handle:
            data = file_handle['data'][slices]

        return data

    def set_data(self, values, slices=None):
        if slices is None:
            slices = slice(None)

        with h5.File(self.path, 'a') as file_handle:
            file_handle['data'][slices] = values


_backend = Backend('hdf5', H5Connection, H5Dataset, H5DataChunk)
