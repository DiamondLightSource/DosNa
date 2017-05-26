

import numpy as np
from collections import namedtuple


# Currently there is no need for more fancy attributes
Backend = namedtuple('Backend', ['name', 'Cluster', 'Pool', 'Dataset', 'DataChunk'])

Engine = namedtuple('Engine', ['name', 'Cluster', 'Pool', 'Dataset', 'DataChunk', 'params'])


class Wrapper(object):
    
    instance = None
    
    def __init__(self, instance):
        self.instance = instance
    
    def __getattr__(self, attr):
        return self.instance.__getattribute__(attr)
    
    def __enter__(self):
        self.instance.__enter__()
        return self
    
    def __exit__(self, *args):
        self.instance.__exit__(*args)


class BaseCluster(object):
    
    def __init__(self, *args, **kwargs):
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
        self._isopen = True
        
    def close(self):
        self._isopen = False
    
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
    
    # Standard implementations, could be overriden for more efficient access
        
    def get_chunk_data(self, idx, slices=None):
        return self.get_chunk(idx)[slices]
    
    def set_chunk_data(self, idx, values, slices=None):
        self.get_chunk(idx)[slices] = values
    
    # To be implemented by Processing Backends
    
    def __getitem__(self, slices):
        return self.get_data(slices=slices)
            
    def __setitem__(self, slices, values):
        return self.set_data(values, slices=slices)
    
    def map(self, func, padding, name):
        raise NotImplementedError('`map` not implemented for this backend')
        
    def apply(self, func, padding):
        raise NotImplementedError('`map` not implemented for this backend')
    
    def load(self, data):
        raise NotImplementedError('`load` not implemented for this backend')
    
    def clone(self, name):
        raise NotImplementedError('`clone` not implemented for this backend')
    
    def get_data(self, slices=None):
        raise NotImplementedError('`get_data` not implemented for this backend')      
    
    def set_data(self, data, slices=None):
        raise NotImplementedError('`set_data` not implemented for this backend')      
    
    # Utility methods used by all backends and engines
    
    def _idx_from_flat(self, idx):
        return tuple(map(int, np.unravel_index(idx, self.chunks)))
    
    def _local_chunk_bounds(self, idx):
         return tuple((slice(0, min((i + 1) * s, self.shape[j]) - i * s)
                      for j, (i, s) in enumerate(zip(idx, self.chunk_size))))
    
    def _global_chunk_bounds(self, idx):
        return tuple((slice(i * s, min((i + 1) * s, self.shape[j]))
                      for j, (i, s) in enumerate(zip(idx, self.chunk_size))))

    def _process_slices(self, slices, squeeze=False):
        if type(slices) in [slice, int]:
            slices = [slices]
        elif slices is Ellipsis:
            slices = [slice(None)]
        elif type(slices) not in [list, tuple]:
            raise Exception('Invalid Slicing with index of type `{}`'.format(type(slices)))
        else:
            slices = list(slices)
        
        if len(slices) <= self.ndim:
            nmiss = self.ndim - len(slices)
            while Ellipsis in slices:
                idx = slices.index(Ellipsis)
                slices = slices[:idx] + ([slice(None)] * (nmiss+1)) + slices[idx+1:]
            if len(slices) < self.ndim:
                slices = list(slices) + ([slice(None)] * nmiss)
        elif len(slices) > self.ndim:
            raise Exception('Invalid slicing of dataset of dimension `{}`'
                            ' with {}-dimensional slicing'
                            .format(self.ndim, len(slices)))
        final_slices = []
        shape = self.shape
        squeeze_axis = []
        for i, s in enumerate(slices):
            if type(s) == int:
                final_slices.append(slice(s, s+1))
                squeeze_axis.append(i)
            elif type(s) == slice:
                start = s.start
                stop = s.stop
                if start is None:
                    start = 0
                if stop is None:
                    stop = shape[i]
                elif stop < 0:
                    stop = self.shape[i] + stop
                if start < 0 or start >= self.shape[i]:
                    raise Exception('Only possitive and in-bounds slicing supported: `{}`'
                                           .format(slices))
                if stop < 0 or stop > self.shape[i] or stop < start:
                    raise Exception('Only possitive and in-bounds slicing supported: `{}`'
                                           .format(slices))
                if s.step is not None and s.step != 1:
                    raise Exception('Only slicing with step 1 supported')
                final_slices.append(slice(start, stop))
            else:
                raise Exception('Invalid type `{}` in slicing, only integer or'
                                ' slices are supported'.format(type(s)))

        if squeeze:
            return final_slices, squeeze_axis
        return final_slices
    
    def _chunk_slice_iterator(self, slices, ndim):
        indexes = []
        ltargets = []
        gtargets = []
        for slice_axis, chunk_axis_size, max_chunks in zip(slices, self.chunk_size, self.chunks):
            start_chunk = slice_axis.start // chunk_axis_size
            end_chunk = min((slice_axis.stop-1) // chunk_axis_size, max_chunks-1)
            pad_start = slice_axis.start - start_chunk * chunk_axis_size
            pad_stop = slice_axis.stop - max(0, end_chunk) * chunk_axis_size
            ltarget = []
            gtarget = []
            index = []
            for i in range(start_chunk, end_chunk+1):
                start = pad_start if i == start_chunk else 0
                stop = pad_stop if i == end_chunk else chunk_axis_size
                ltarget.append(slice(start, stop))
                gchunk = i * chunk_axis_size - slice_axis.start
                gtarget.append(slice(gchunk + start, gchunk + stop))
                index.append(i)
            ltargets.append(ltarget)
            gtargets.append(gtarget)
            indexes.append(index)

        def __chunk_iterator():
            for idx in np.ndindex(*[len(chunks_axis) for chunks_axis in indexes]):
                _index = []; _lslice = []; _gslice = []
                for n, j in enumerate(idx):
                    _index.append(indexes[n][j])
                    _lslice.append(ltargets[n][j])
                    if self.ndim - ndim <= n:
                        _gslice.append(gtargets[n][j])
                yield tuple(_index), tuple(_lslice), tuple(_gslice)

        return __chunk_iterator()


class BaseDataChunk(object):
    
    def __init__(self, dataset, idx, name, shape, dtype, fillvalue):
        if not dataset.has_chunk(idx):
            raise Exception('Wrong initialization of a DataChunk')
        self._dataset = dataset
        self._idx = idx
        self._name = name
        self._shape = shape
        self._dtype = dtype
        self._fillvalue = fillvalue
    
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
        return self.get_data(slices=slices)

    def __setitem__(self, slices, values):
        self.set_data(values, slices=slices)
