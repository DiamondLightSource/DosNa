#!/usr/bin/env python
"""backend RAM keeps every data structure in memory"""

import logging

import numpy as np

from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset, DatasetNotFoundError)

log = logging.getLogger(__name__)


class MemConnection(BackendConnection):
    """
    A Memory Connection represents a dictionary.
    """

    def __init__(self, *args, **kwargs):
        super(MemConnection, self).__init__(*args, **kwargs)
        self.datasets = {}
        self.trees = {}
        
    def create_tree(self, name):
        self.trees[name] = {}
        backendtree = MemTree(self, name)
        self.trees[name][name] = backendtree
        return backendtree
    
    def get_tree(self, name):
        if not self.has_tree(name):
            raise BackendTreeNotFoundError("Backend `%s` does not exist")
        return self.trees[name][name]

    def has_tree(self, name):
        return name in self.trees

    def del_tree(self, name):
        if not self.has_dataset(name):
            raise BackendTreeNotFoundError("Backend `%s` does not exist")
        log.debug("Removing Backend `%s`", name)
        del self.trees[name]

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

        log.debug('Creating Dataset `%s`', name)
        self.datasets[name] = None  # Key `name` has to exist
        dataset = MemDataset(self, name, shape, dtype, fillvalue, chunk_grid,
                             chunk_size)
        self.datasets[name] = dataset
        return dataset

    def get_dataset(self, name):
        if not self.has_dataset(name):
            raise DatasetNotFoundError('Dataset `%s` does not exist' % name)
        return self.datasets[name]

    def has_dataset(self, name):
        return name in self.datasets

    def del_dataset(self, name):
        if not self.has_dataset(name):
            raise DatasetNotFoundError('Dataset `%s` does not exist' % name)
        log.debug('Removing Dataset `%s`', name)
        del self.datasets[name]
        
class MemTree: # TODO: add the BackendTree
    """
    A Memory Tree represents a dictionary of dictionaries
    """
    
    def __init__(self, connection_handler, name, *args, **kwargs):
        # super(MemTree, self).__init__(*args, **kwargs)
        self.connection_handler = connection_handler
        self.name = name
        self.metadata = {}
        self.datasets = {}
        
    def create_tree(self, name, connection_handler):
        log.debug("Creating Tree `%s`", name)
        tree = MemTree(self, name)
        """
        def _recurse(treeobject):
            for key, value in treeobject.items():
                if key == self.name:
                    treeobject[self.name][treename] = {}
                    for k, v in value.items():
                        if k == treename:
                            value[treename] = {}
                            value[treename][treename] = tree
                else:
                    if isinstance(value, dict):
                        _recurse(value)
            return treeobject

        datadict = _recurse(connection.trees)
        """
        return tree
    
    def create_dataset(
        self,
        name,
        shape=None,
        dtype=np.float32,
        fillvalue=0,
        data=None,
        chunk_size=None,
    ):

        if not ((shape is not None and dtype is not None) or data is not None):
            raise Exception("Provide `shape` and `dtype` or `data`")
        if self.has_dataset(name):
            raise Exception("Dataset `%s` already exists" % name)

        if data is not None:
            shape = data.shape
            dtype = data.dtype

        if chunk_size is None:
            chunk_size = shape

        chunk_grid = (np.ceil(np.asarray(shape, float) / chunk_size)).astype(int)

        log.debug("Creating Dataset `%s`", name)
        self.datasets[name] = None  # Key `name` has to exist
        dataset = MemDataset(
            self, name, shape, dtype, fillvalue, chunk_grid, chunk_size
        )
        self.datasets[name] = dataset
        return dataset

    def get_dataset(self, name):
        if not self.has_dataset(name):
            raise DatasetNotFoundError("Dataset `%s` does not exist")
        return self.datasets[name]

    def has_dataset(self, name):
        return name in self.datasets

    def del_dataset(self, name):
        if not self.has_dataset(name):
            raise DatasetNotFoundError("Dataset `%s` does not exist")
        log.debug("Removing Dataset `%s`", name)
        del self.datasets[name]


class MemDataset(BackendDataset):

    def __init__(self, pool, name, shape, dtype, fillvalue, chunk_grid,
                 chunk_size):
        super(MemDataset, self).__init__(pool, name, shape, dtype, fillvalue,
                                         chunk_grid, chunk_size)
        self.data_chunks = {}
        self._populate_chunks()

    def _populate_chunks(self):
        for idx in np.ndindex(*self.chunk_grid):
            self.create_chunk(idx)

    def create_chunk(self, idx, data=None, slices=None):
        if self.has_chunk(idx):
            raise Exception('DataChunk `{}{}` already exists'.format(self.name,
                                                                     idx))

        self.data_chunks[idx] = None

        chunk = MemDataChunk(self, idx, 'Chunk {}'.format(idx),
                             self.chunk_size, self.dtype, self.fillvalue)
        if data is not None:
            slices = slices or slice(None)
            chunk.set_data(data, slices=slices)

        self.data_chunks[idx] = chunk
        return chunk

    def get_chunk(self, idx):
        if self.has_chunk(idx):
            return self.data_chunks[idx]
        return self.create_chunk(idx)

    def has_chunk(self, idx):
        return idx in self.data_chunks

    def del_chunk(self, idx):
        if self.has_chunk(idx):
            del self.data_chunks[idx]
            return True
        return False


class MemDataChunk(BackendDataChunk):

    def __init__(self, dataset, idx, name, shape, dtype, fillvalue):
        super(MemDataChunk, self).__init__(dataset, idx, name, shape,
                                           dtype, fillvalue)
        self.data = np.full(shape, fillvalue, dtype)

    def get_data(self, slices=None):
        return self.data[slices]

    def set_data(self, values, slices=None):
        self.data[slices] = values


_backend = Backend('ram', MemConnection, MemDataset, MemDataChunk)
