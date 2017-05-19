

import numpy as np


class FreezeConstructor(type):

    def __new__(metacls, name, bases, dct):
         if "__init__" in dct:
              raise NameError('Backend methods shouldnt have an init method')
         return type.__new__(metacls, name, bases, dct)


class BaseCluster(object):
    
    def __init__(self):
        self._connected = False
    
    @property
    def connected(self):
        return self._connected
    
    def connect(self):
        self._connected = True
    
    def disconnect(self):
        self._connected = True
    
    def __enter__(self):
        if not self.connected:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connected:
            self.disconnect()
            
    def create_pool(self, name, open_mode='a'):
        raise NotImplementedError('`create_pool` not implemented for this backend')
    
    def get_pool(self, name, open_mode='a'):
        raise NotImplementedError('`get_pool` not implemented for this backend')
    
    def has_pool(self, name):
        raise NotImplementedError('`has_pool` not implemented for this backend')
    
    def del_pool(self, name):
        raise NotImplementedError('`delete_pool` not implemented for this backend')
    
    def __getitem__(self, name):
        return self.get_pool(name)
    
    def __contains__(self, name):
        return self.has_pool(name)


class BasePool(object):
    
    __metaclass__ = FreezeConstructor
    
    def __init__(self, cluster, name, open_mode='a'):
        if not cluster.has_pool(name):
            raise Exception('Wrong initialization of a Pool')
        
        self._cluster = cluster
        self._name = name
        self._mode = open_mode
        self._isopen = False
        
    @property
    def name(self):
        return self._name
    
    @property
    def isopen(self):
        return self._isopen
    
    @property
    def mode(self):
        return self._mode
        
    def open(self):
        raise NotImplementedError('`open` not implemented for this backend')
        
    def close(self):
        raise NotImplementedError('`close` not implemented for this backend')
    
    def __enter__(self):
        if not self.isopen:
            self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.isopen:
            self.close()
    
    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0, 
                       data=None, chunks=None):
        raise NotImplementedError('`create_dataset` not implemented for this backend')
    
    def get_dataset(self, name):
        raise NotImplementedError('`get_dataset` not implemented for this backend')
    
    def has_dataset(self, name):
        raise NotImplementedError('`has_dataset` not implemented for this backend')
    
    def del_dataset(self, name):
        raise NotImplementedError('`del_dataset` not implemented for this backend')
    
    def __getitem__(self, name):
        return self.get_dataset(name)
    
    def __contains__(self, name):
        return self.has_dataset(name)
        

class BaseDataset(object):
    
    __metaclass__ = FreezeConstructor
    
    def __init__(self, pool, name, shape, dtype, fillvalue, chunks, chunk_size):
        if not pool.has_dataset(name):
            raise Exception('Wrong initialization of a Dataset')
            
        self._pool = pool
        self._name = name
        self._shape = shape
        self._dtype = dtype
        self._fillvalue = fillvalue

        self._chunks = chunks
        self._chunk_size = chunk_size
        self._total_chunks = np.prod(chunks)
    
    @property
    def name(self):
        return self._name
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def ndim(self):
        return len(self._shape)
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def fillvalue(self):
        return self._fillvalue
    
    @property
    def chunks(self):
        return self._chunks
    
    @property
    def chunk_size(self):
        return self._chunk_size
    
    @property
    def total_chunks(self):
        return self._total_chunks
    
    # To be implementd by Storage Backend
    
    def create_chunk(self, idx, data=None, cslices=None):
        raise NotImplementedError('`create_chunk` not implemented for this backend')
    
    def get_chunk(self, idx):
        raise NotImplementedError('`get_chunk` not implemented for this backend')        
        
    def has_chunk(self, idx):
        raise NotImplementedError('`has_chunk` not implemented for this backend')
        
    def del_chunk(self, idx):
        raise NotImplementedError('`del_chunk` not implemented for this backend')
        
    def get_chunk_data(self, idx):
        raise NotImplementedError('`get_chunk_data` not implemented for this backend')
    
    def set_chunk_data(self, idx):
        raise NotImplementedError('`set_chunk_data` not implemented for this backend')
    
    # To be implemented by Processing Backends
    
    def __getitem__(self, slices):
        raise NotImplementedError('`slicing` not implemented for this backend')
            
    def __setitem__(self, slices):
        raise NotImplementedError('`slicing` not implemented for this backend')
    
    def map(self, func, padding):
        raise NotImplementedError('`map` not implemented for this backend')
        
    def apply(self, func, padding):
        raise NotImplementedError('`map` not implemented for this backend')
    
    def load(self, data):
        raise NotImplementedError('`load` not implemented for this backend')


class BaseDataChunk(object):
    
    __metaclass__ = FreezeConstructor
    
    def __init__(self, dataset, name, shape, dtype):
        if not self.dataset.has_chunk(name):
            raise Exception('Wrong initialization of a DataChunk')
        self._name = name
        self._shape = shape
        self._dtype = dtype
    
    @property
    def name(self):
        return self._name
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def fillvalue(self):
        return self._fillvalue

    def get_data(self, slices=None):
        raise NotImplementedError('`get_data` not implemented for this backend')
    
    def set_data(self, values, slices=None):
        raise NotImplementedError('`set_data` not implemented for this backend')
    
    def __getitem__(self, slices):
        return self.get_slices(slices=slices)

    def __setitem__(self, slices, values):
        self.set_slices(values, slices=slices)
