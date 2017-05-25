

import os
import shutil

import logging as log

import h5py as h5
import numpy as np

from .. import Backend
from ..base import BaseCluster, BasePool, BaseDataset, BaseDataChunk
from ..utils import dtype2str


class H5Cluster(BaseCluster):
    """
    A HDF5 Cluster represents the local filesystem.
    """
    
    def __init__(self, path):
        super(H5Cluster, self).__init__()
        self.path = ''
        self.path = self._validate_path(path)
        
    def connect(self):
        super(H5Cluster, self).connect()
        log.debug('Starting HDF5 Cluster at `%s`' % self.path)
    
    def disconnect(self):
        super(H5Cluster, self).disconnect()
        log.debug('Stopping HDF5 Cluster at `%s`' % self.path)
    
    def _validate_path(self, path):
        path = os.path.realpath(os.path.join(self.path, path))
        if len(os.path.splitext(path)[1]) > 0:
            raise Exception('`%s` is not a valid Pool path' % path)
        return path
    
    def create_pool(self, path, open_mode='a'):
        path = self._validate_path(path)
        if os.path.exists(path):
            raise Exception('Path `%s` already exists' % path)
        
        log.debug('Creating pool `%s`' % path)
        
        os.makedirs(path)
        flag = os.path.join(path, '.dosna')
        with open(os.path.join(path, '.dosna'), 'w'):
            os.utime(flag, None)
        return H5Pool(self, path, open_mode=open_mode)
    
    def get_pool(self, path, open_mode='a'):
        path = self._validate_path(path)
        if self.has_pool(path):
            return H5Pool(self, path, open_mode=open_mode)
        if os.path.exists(path):
            raise Exception('Path `%s` is not a pool' % path)
        raise Exception('Pool `%s` does not exist' % path)
    
    def has_pool(self, path):
        path = self._validate_path(path)
        return os.path.isdir(path) and os.path.isfile(os.path.join(path, '.dosna'))
    
    def del_pool(self, path):
        real_path = self._validate_path(path)
        if self.has_pool(real_path):
            log.debug('Removing pool `%s`' % real_path)
            shutil.rmtree(real_path)
        if os.path.exists(real_path):
            raise Exception('Path `%s` is not a valid pool' % real_path)
     
        
class H5Pool(BasePool):
    """
    An HDF5 Pool represents a subdirectory in the local filesystem with a `.dosna` file.
    """
    
    def _process_path(self, path):
        return  os.path.realpath(os.path.join(self.name, path))
    
    def create_dataset(self, path, shape=None, dtype=np.float32, fillvalue=0, 
                       data=None, chunks=None):
        
        real_path = self._process_path(path)
        
        if not ((shape is not None and dtype is not None) or data is not None):
            raise Exception('Provide `shape` and `dtype` or `data`')
        if self.has_dataset(path):
            raise Exception('Dataset `%s` already exists' % path)

        if data is not None:
            shape = data.shape
            dtype = data.dtype
        
        if chunks is None:
            chunk_size = shape
        else:
            chunk_size = chunks
        chunks_needed = (np.ceil(np.asarray(shape, float) / chunk_size)).astype(int)
        
        os.mkdir(real_path)
        with h5.File(os.path.join(real_path, 'dataset.h5'), 'w') as f:
            f.attrs['shape'] = shape
            f.attrs['dtype'] = dtype2str(dtype)
            f.attrs['fillvalue'] = np.dtype(dtype).type(fillvalue)
            f.attrs['chunks'] = np.asarray(chunks_needed, dtype=int)
            f.attrs['chunk_size'] = np.asarray(chunk_size, dtype=int)
        
        log.debug('Creating dataset `%s`' % real_path)
        
        dataset = H5Dataset(self, real_path, shape, dtype, fillvalue, 
                            chunks_needed, chunk_size)
        
        return dataset
    
    def get_dataset(self, path):
        real_path = self._process_path(path)
        
        if not self.has_dataset(path):
            raise Exception('Dataset `%s` does not exist' % path)
            
        with h5.File(os.path.join(real_path, 'dataset.h5'), 'w') as f:
            shape = f.attrs['shape']
            dtype = f.attrs['dtype']
            fillvalue = f.attrs['fillvalue']
            chunks_needed = f.attrs['chunks']
            chunk_size = f.attrs['chunk_size']
        
        return H5Dataset(self, path, shape, dtype, fillvalue, chunks_needed, chunk_size)
    
    def has_dataset(self, path):
        real_path = self._process_path(path)
        return os.path.isdir(real_path) and os.path.isfile(os.path.join(real_path, 'dataset.h5'))
    
    def del_dataset(self, path):
        real_path = self._process_path(path)
        if not self.has_dataset(path):
            raise Exception('Dataset `%s` does not exist' % path)
        log.debug('Removing Dataset `%s`' % os.path.join(real_path, path))
        shutil.rmtree(real_path)
            
        
class H5Dataset(BaseDataset):
    
    def _idx2name(self, idx):
        if not all([type(i) == int for i in idx]) or len(idx) != self.ndim:
            raise Exception('Invalid chunk idx')
        return os.path.join(self.name, 'chunk_%s.h5' % '_'.join(map(str, idx)))
    
    def create_chunk(self, idx, data=None, slices=None):
        if self.has_chunk(idx):
            raise Exception('DataChunk `{}` already exists'.format(idx))
        
        chunk_name = self._idx2name(idx)
        with h5.File(chunk_name, 'w') as f:
            f.create_dataset('data', shape=self.chunk_size, dtype=self.dtype,
                             fillvalue=self.fillvalue)
            if data is not None:
                slices = slices or slice(None)
                f['data'][slices] = data

        return H5DataChunk(self, idx, chunk_name, self.chunk_size, self.dtype,
                           self.fillvalue)
    
    def get_chunk(self, idx):
        if self.has_chunk(idx):
            name = self._idx2name(idx)
            with h5.File(name, 'r') as f:
                shape = f['data'].shape
                dtype = f['data'].dtype
            return H5DataChunk(self, idx, name, shape, dtype, self.fillvalue)
        return self.create_chunk(idx)
        
    def has_chunk(self, idx):
        return os.path.isfile(self._idx2name(idx))
        
    def del_chunk(self, idx):
        if not self.has_chunk(idx):
            raise Exception('DataChunk `{}` does not exist'.format(idx))
        os.remove(self._idx2name(idx))


class H5DataChunk(BaseDataChunk):
    
    def get_data(self, slices=None):
        if slices is None:
            slices = slice(None)

        with h5.File(self.name, 'r') as f:
            data = f['data'][slices]

        return data

    def set_data(self, values, slices=None):
        if slices is None:
            slices = slice(None)

        with h5.File(self.name, 'a') as f:
            f['data'][slices] = values


backend = Backend('hdf5', H5Cluster, H5Pool, H5Dataset, H5DataChunk)
