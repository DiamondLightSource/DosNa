

import numpy as np
import logging as log

from .. import Backend
from ..base import BaseCluster, BasePool, BaseDataset, BaseDataChunk


class MemCluster(BaseCluster):
    """
    A Memory Cluster represents a dictionary.
    """
    
    def __init__(self, path):
        super(MemCluster, self).__init__()
        self.pools = {}
        
    def connect(self):
        super(MemCluster, self).connect()
        log.debug('Starting Memory Cluster')
    
    def disconnect(self):
        super(MemCluster, self).disconnect()
        log.debug('Stopping Memory Cluster')
    
    def create_pool(self, name, open_mode='a'):
        if self.has_pool(name):
            raise Exception('Path `%s` already exists' % name)
        
        log.debug('Creating pool `%s`' % name)
        self.pools[name] = None # Key `name` has to exist before instantiating pool
        pool = MemPool(self, name, open_mode=open_mode)
        self.pools[name] = pool
        return pool
    
    def get_pool(self, name, open_mode='a'):
        if self.has_pool(name):
            return self.pools[name]
        raise Exception('Pool `%s` does not exist' % name)
    
    def has_pool(self, name):
        return name in self.pools
    
    def del_pool(self, name):
        if self.has_pool(name):
            log.debug('Removing pool `%s`' % name)
            del self.pools[name]
        else:
            raise Exception('Pool `%s` does not exist' % name)
     
        
class MemPool(BasePool):
    """
    An Mem Pool represents a dictionary.
    """
    
    def __init__(self, cluster, name, open_mode='a'):
        super(MemPool, self).__init__(cluster, name, open_mode=open_mode)
        self.datasets = {}
        
    def open(self):
        pass
    
    def close(self):
        pass
    
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
            chunk_size = shape
        else:
            chunk_size = chunks
        chunks_needed = (np.ceil(np.asarray(shape, float) / chunk_size)).astype(int)
        
        log.debug('Creating Dataset `%s`' % name)
        self.datasets[name] = None # Key `name` has to exist
        dataset = MemDataset(self, name, shape, dtype, fillvalue, 
                             chunks_needed, chunk_size)
        self.datasets[name] = dataset
        return dataset
    
    def get_dataset(self, name):        
        if self.has_dataset(name):
            raise Exception('Dataset `%s` does not exist' % name)
        return self.datasets[name]
    
    def has_dataset(self, name):
        return name in self.datasets
    
    def del_dataset(self, name):
        if not self.has_dataset(name):
            raise Exception('Dataset `%s` does not exist' % name)
        log.debug('Removing Dataset `%s`' % name)
        del self.datasets[name]
            
        
class MemDataset(BaseDataset):
    
    def __init__(self, pool, name, shape, dtype, fillvalue, chunks, chunk_size):
        super(MemDataset, self).__init__(pool, name, shape, dtype, fillvalue, 
                                         chunks, chunk_size)
        self.data_chunks = {}
    
    def create_chunk(self, idx, data=None, slices=None):
        if self.has_chunk(idx):
            raise Exception('DataChunk `{}` already exists'.format(idx))
        
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
        if not self.has_chunk(idx):
            raise Exception('DataChunk `{}` does not exist'.format(idx))
        del self.data_chunks[idx]


class MemDataChunk(BaseDataChunk):
    
    def __init__(self, dataset, idx, name, shape, dtype, fillvalue):
        super(MemDataChunk, self).__init__(dataset, idx, name, shape, dtype, fillvalue)
        self.data = np.full(shape, fillvalue, dtype)
    
    def get_data(self, slices=None):
        return self.data[slices]

    def set_data(self, values, slices=None):
        self.data[slices] = values


backend = Backend('memory', MemCluster, MemPool, MemDataset, MemDataChunk)
