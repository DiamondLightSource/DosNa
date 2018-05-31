#!/usr/bin/env python
"""Base classes for every backend"""

import logging
from itertools import product

import numpy as np

from six.moves import range

log = logging.getLogger(__name__)


class BackendConnection(object):

    def __init__(self, name, open_mode="a", *args, **kwargs):
        self._name = name
        self._connected = False
        self._mode = open_mode
        log.debug('Extra connection options: args=%s kwargs=%s', args, kwargs)

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
                       data=None, chunk_size=None):
        raise NotImplementedError('`create_dataset` not implemented '
                                  'for this backend')

    def get_dataset(self, name):
        raise NotImplementedError('`get_dataset` not implemented '
                                  'for this backend')

    def has_dataset(self, name):
        raise NotImplementedError('`has_dataset` not implemented '
                                  'for this backend')

    def del_dataset(self, name):
        """Remove dataset metadata only"""
        raise NotImplementedError('`del_dataset` not implemented '
                                  'for this backend')


class BackendDataset(object):

    def __init__(self, connection, name, shape, dtype, fillvalue, chunk_grid,
                 chunk_size):
        if not connection.has_dataset(name):
            raise Exception('Wrong initialization of a Dataset')

        self._connection = connection
        self._name = name
        self._shape = shape
        self._dtype = dtype
        self._fillvalue = fillvalue

        self._chunk_grid = chunk_grid
        self._chunk_size = chunk_size
        self._total_chunks = np.prod(chunk_grid)
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
    def chunk_grid(self):
        return self._chunk_grid

    @property
    def chunk_size(self):
        return self._chunk_size

    @property
    def total_chunks(self):
        return self._total_chunks

    # To be implementd by Storage Backend

    def create_chunk(self, idx, data=None, slices=None):
        raise NotImplementedError('`create_chunk` not implemented '
                                  'for this backend')

    def get_chunk(self, idx):
        raise NotImplementedError('`get_chunk` not implemented '
                                  'for this backend')

    def has_chunk(self, idx):
        raise NotImplementedError('`has_chunk` not implemented '
                                  'for this backend')

    def del_chunk(self, idx):
        raise NotImplementedError('`del_chunk` not implemented '
                                  'for this backend')

    # Standard implementations, could be overriden for more efficient access

    def get_chunk_data(self, idx, slices=None):
        return self.get_chunk(idx)[slices]

    def set_chunk_data(self, idx, values, slices=None):
        if not self.has_chunk(idx):
            self.create_chunk(idx, values, slices)
        self.get_chunk(idx)[slices] = values

    # Utility methods used by all backends and engines

    def _idx_from_flat(self, idx):
        return tuple(map(int, np.unravel_index(idx, self.chunk_grid)))

    def _local_chunk_bounds(self, idx):
        return tuple((slice(0, min((i + 1) * s, self.shape[j]) - i * s)
                      for j, (i, s) in enumerate(zip(idx, self.chunk_size))))

    def _global_chunk_bounds(self, idx):
        return tuple((slice(i * s, min((i + 1) * s, self.shape[j]))
                      for j, (i, s) in enumerate(zip(idx, self.chunk_size))))

    def _process_slices(self, slices, squeeze=False):
        if isinstance(slices, (slice, int)):
            slices = [slices]
        elif slices is Ellipsis:
            slices = [slice(None)]
        elif np.isscalar(slices):
            slices = [int(slices)]
        elif not isinstance(slices, (list, tuple)):
            raise Exception('Invalid Slicing with index of type `{}`'
                            .format(type(slices)))
        else:
            slices = list(slices)

        if len(slices) <= self.ndim:
            nmiss = self.ndim - len(slices)
            while Ellipsis in slices:
                idx = slices.index(Ellipsis)
                slices = slices[:idx] + ([slice(None)] * (nmiss + 1)) \
                    + slices[idx + 1:]
            if len(slices) < self.ndim:
                slices = list(slices) + ([slice(None)] * nmiss)
        elif len(slices) > self.ndim:
            raise Exception('Invalid slicing of dataset of dimension `{}`'
                            ' with {}-dimensional slicing'
                            .format(self.ndim, len(slices)))
        final_slices = []
        shape = self.shape
        squeeze_axis = []
        for index, slice_ in enumerate(slices):
            if isinstance(slice_, int):
                final_slices.append(slice(slice_, slice_ + 1))
                squeeze_axis.append(index)
            elif isinstance(slice_, slice):
                start = slice_.start
                stop = slice_.stop
                if start is None:
                    start = 0
                if stop is None:
                    stop = shape[index]
                elif stop < 0:
                    stop = self.shape[index] + stop
                if start < 0 or start >= self.shape[index]:
                    raise Exception('Only possitive and '
                                    'in-bounds slicing supported: `{}`'
                                    .format(slices))
                if stop < 0 or stop > self.shape[index] or stop < start:
                    raise Exception('Only possitive and '
                                    'in-bounds slicing supported: `{}`'
                                    .format(slices))
                if slice_.step is not None and slice_.step != 1:
                    raise Exception('Only slicing with step 1 supported')
                final_slices.append(slice(start, stop))
            else:
                raise Exception('Invalid type `{}` in slicing, only integer or'
                                ' slices are supported'.format(type(slice_)))

        if squeeze:
            return final_slices, squeeze_axis
        return final_slices

    @staticmethod
    def _ndindex(dims):
        return product(*(range(d) for d in dims))

    def _chunk_slice_iterator(self, slices, ndim):
        indexes = []
        nchunks = []
        cslices = []
        gslices = []

        chunk_size = self.chunk_size
        chunk_grid = self.chunk_grid

        for index, slc in enumerate(slices):
            sstart = slc.start // chunk_size[index]
            sstop = min((slc.stop - 1) // chunk_size[index],
                        chunk_grid[index] - 1)
            if sstop < 0:
                sstop = 0

            pad_start = slc.start - sstart * chunk_size[index]
            pad_stop = slc.stop - sstop * chunk_size[index]

            _i = []  # index
            _c = []  # chunk slices in current dimension
            _g = []  # global slices in current dimension

            for chunk_index in range(sstart, sstop + 1):
                start = pad_start if chunk_index == sstart else 0
                stop = pad_stop if chunk_index == sstop else chunk_size[index]
                gchunk = chunk_index * chunk_size[index] - slc.start
                _i += [chunk_index]
                _c += [slice(start, stop)]
                _g += [slice(gchunk + start, gchunk + stop)]

            nchunks += [sstop - sstart + 1]
            indexes += [_i]
            cslices += [_c]
            gslices += [_g]

        return (zip(*
                    (
                        (
                            indexes[n][i],
                            cslices[n][i],
                            (n < ndim or None) and gslices[n][i],
                        )
                        for n, i in enumerate(idx)
                    ))
                for idx in self._ndindex(nchunks))


class BackendDataChunk(object):

    def __init__(self, dataset, idx, name, shape, dtype, fillvalue):
        self._dataset = dataset
        self._idx = idx
        self._name = name
        self._shape = shape
        self._size = np.prod(shape)
        self._dtype = dtype
        self._fillvalue = fillvalue
        self._byte_count = self.size * np.dtype(self.dtype).itemsize

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

    @property
    def byte_count(self):
        return self._byte_count

    def get_data(self, slices=None):
        raise NotImplementedError('`get_data` not implemented '
                                  'for this backend')

    def set_data(self, values, slices=None):
        raise NotImplementedError('`set_data` not implemented '
                                  'for this backend')

    def __getitem__(self, slices):
        return self.get_data(slices=slices)

    def __setitem__(self, slices, values):
        self.set_data(values, slices=slices)


class ConnectionError(Exception):
    pass


class DatasetNotFoundError(Exception):
    pass
