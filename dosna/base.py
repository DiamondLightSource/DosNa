

from collections import namedtuple
from itertools import product
import logging

import numpy as np
from six.moves import range

log = logging.getLogger(__name__)

# Currently there is no need for more fancy attributes
Backend = namedtuple('Backend', ['name', 'Connection', 'Dataset', 'DataChunk'])

Engine = namedtuple('Engine', ['name', 'Connection', 'Dataset', 'DataChunk', 'params'])


class Wrapper(object):

    instance = None

    def __init__(self, instance):
        self.instance = instance

    def __getattr__(self, attr):
        """
        Attributes/Functions that do not exist in the extended class
        are going to be passed to the instance being wrapped
        """
        return self.instance.__getattribute__(attr)

    def __enter__(self):
        self.instance.__enter__()
        return self

    def __exit__(self, *args):
        self.instance.__exit__(*args)


class BaseConnection(object):

    def __init__(self, name, open_mode="a", *args, **kwargs):
        self._name = name
        self._connected = False
        self._mode = open_mode

    @property
    def name(self):
        return self._name

    @property
    def connected(self):
        return self._connected

    @property
    def mode(self):
        return self._mode

    def connect(self):
        log.debug("Connecting to %s", self.name)
        self._connected = True

    def disconnect(self):
        log.debug("Disconnecting from %s", self.name)
        self._connected = False

    def __enter__(self):
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connected:
            self.disconnect()

    def __getitem__(self, name):
        return self.get_dataset(name)

    def __contains__(self, name):
        return self.has_dataset(name)

    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunks=None):
        raise NotImplementedError('`create_dataset` not implemented for this backend')

    def get_dataset(self, name):
        raise NotImplementedError('`get_dataset` not implemented for this backend')

    def has_dataset(self, name):
        raise NotImplementedError('`has_dataset` not implemented for this backend')

    def del_dataset(self, name):
        raise NotImplementedError('`del_dataset` not implemented for this backend')


class BaseDataset(object):

    def __init__(self, connection, name, shape, dtype, fillvalue, chunks, csize):
        if not connection.has_dataset(name):
            raise Exception('Wrong initialization of a Dataset')

        self._connection = connection
        self._name = name
        self._shape = shape
        self._dtype = dtype
        self._fillvalue = fillvalue

        self._chunks = chunks
        self._chunk_size = csize
        self._total_chunks = np.prod(chunks)
        self._ndim = len(self._shape)

    @property
    def connection(self):
        return self._connection


    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._ndim

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

    def clear(self):
        raise NotImplementedError('`clear` not implemented for this backend')

    def delete(self):
        raise NotImplementedError('`delete` not implemented for this backend')

    def map(self, func, output_name):
        raise NotImplementedError('`map` not implemented for this backend')

    def apply(self, func):
        raise NotImplementedError('`apply` not implemented for this backend')

    def load(self, data):
        raise NotImplementedError('`load` not implemented for this backend')

    def clone(self, output_name):
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
        elif np.isscalar(slices):
            slices = [int(slices)]
        elif type(slices) not in [list, tuple]:
            raise Exception('Invalid Slicing with index of type `{}`'
                            .format(type(slices)))
        else:
            slices = list(slices)

        if len(slices) <= self.ndim:
            nmiss = self.ndim - len(slices)
            while Ellipsis in slices:
                idx = slices.index(Ellipsis)
                slices = slices[:idx] + ([slice(None)] * (nmiss + 1)) + slices[idx + 1:]
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
                final_slices.append(slice(s, s + 1))
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

    def _ndindex(self, dims):
        return product(*(range(d) for d in dims))

    def _chunk_slice_iterator(self, slices, ndim):
        indexes = []
        nchunks = []
        cslices = []
        gslices = []

        chunk_size = self.chunk_size
        chunks = self.chunks

        for n, slc in enumerate(slices):
            sstart = slc.start // chunk_size[n]
            sstop = min((slc.stop - 1) // chunk_size[n], chunks[n] - 1)
            if sstop < 0:
                sstop = 0

            pad_start = slc.start - sstart * chunk_size[n]
            pad_stop = slc.stop - sstop * chunk_size[n]

            _i = []  # index
            _c = []  # chunk slices in current dimension
            _g = []  # global slices in current dimension

            for i in range(sstart, sstop + 1):
                start = pad_start if i == sstart else 0
                stop = pad_stop if i == sstop else chunk_size[n]
                gchunk = i * chunk_size[n] - slc.start
                _i += [i]
                _c += [slice(start, stop)]
                _g += [slice(gchunk + start, gchunk + stop)]

            nchunks += [sstop - sstart + 1]
            indexes += [_i]
            cslices += [_c]
            gslices += [_g]

        return (
            zip(*
                (
                    (
                        indexes[n][i],
                        cslices[n][i],
                        (n < ndim or None) and gslices[n][i],
                    )
                    for n, i in enumerate(idx)
                )
            )
            for idx in self._ndindex(nchunks)
        )


class BaseDataChunk(object):

    def __init__(self, dataset, idx, name, shape, dtype, fillvalue):
        if not dataset.has_chunk(idx):
            raise Exception('Wrong initialization of a DataChunk')
        self._dataset = dataset
        self._idx = idx
        self._name = name
        self._shape = shape
        self._size = np.prod(shape)
        self._dtype = dtype
        self._fillvalue = fillvalue

    @property
    def dataset(self):
        return self._dataset

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

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
