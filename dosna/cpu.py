

import numpy as np

from .backends import backend


__backend__ = backend()


class Cluster(__backend__.Cluster):
    
    def create_pool(self, name, open_mode='a', **kwargs):
        pool = super(Cluster, self).create_pool(name, open_mode=open_mode, **kwargs)
        return Pool(self, pool.name, pool.mode) # Override with this Pool object
    
    def get_pool(self, name, open_mode='a'):
        pool = super(Cluster, self).get_pool(name, open_mode=open_mode)
        return Pool(self, pool.name, pool.mode) # Override with this Pool object
  

class Pool(__backend__.Pool):

    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0, 
                       data=None, chunks=None):
        ds = super(Pool, self).create_dataset(name, shape, dtype, fillvalue,
                                              data, chunks)
        ds = Dataset(self, ds.name, ds.shape, ds.dtype, ds.fillvalue, 
                     ds.chunks, ds.chunk_size)
        
        if data is not None:
            ds.load(data)
        
        return ds
    
    def get_dataset(self, name):
        ds = super(Pool, self).get_dataset(name)
        return Dataset(self, ds.name, ds.shape, ds.dtype, ds.fillvalue, 
                       ds.chunks, ds.chunk_size)
    

class Dataset(__backend__.Dataset):
    
    def _global_chunk_bounds(self, idx):
         return tuple((slice(0, min((i + 1) * s, self.shape[j]) - i * s)
                      for j, (i, s) in enumerate(zip(idx, self.chunk_size))))
    
    def _local_chunk_bounds(self, idx):
        return tuple((slice(i * s, min((i + 1) * s, self.shape[j]))
                      for j, (i, s) in enumerate(zip(idx, self.chunk_size))))
    
    def load(self, data):
        chunks = self.chunks
        # TODO: Fix chunks
        for idx in np.ndindex(*chunks):
            gslices = self._gchunk_bounds_slices(idx)
            lslices = self._lchunk_bounds_slices(idx)
            self._set_chunk_data(idx, data[gslices], slices=lslices)

DataChunk = __backend__.DataChunk