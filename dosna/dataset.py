

import rados

import itertools
import numpy as np

import tempfile
from joblib import Parallel, delayed, load, dump


from .base import BaseData
from .chunk import DataChunk
from .utils import shape2str, str2shape, dtype2str


class DatasetException(Exception):
    pass


class Dataset(BaseData):

    __signature__ = 'DosNa.Dataset'

    def __init__(self, pool, name, data=None, chunks=None, read_only=False, njobs=None):
        if (data is None or read_only) and not pool.has_dataset(name):
            raise DatasetException('No dataset `{}` found in pool `{}`'
                                   .format(name, pool.name))
        elif data is not None:
            Dataset.create(pool, name, data=data, chunks=chunks)

        super(Dataset, self).__init__(pool, name, read_only)

        self._fillvalue = np.dtype(self.dtype).type(self.get_xattr('fillvalue'))
        self._chunks = str2shape(self.get_xattr('chunks'))
        self._chunk_size = str2shape(self.get_xattr('chunk_size'))
        self._chunk_bytes = np.prod(self._chunk_size) * self.itemsize
        self._total_chunks = np.prod(self._chunks)
        self._njobs = pool.njobs if njobs is None else njobs

    ###########################################################
    # VALIDATE SLICING
    ###########################################################

    def _process_slices(self, slices, detect_squeeze=False):
        # Support single axis slicing
        if type(slices) in [slice, int]:
            slices = [slices]
        elif slices is Ellipsis:
            slices = [slice(None)]
        elif type(slices) not in [list, tuple]:
            raise DatasetException('Invalid Slicing with index of type `{}`'.format(type(slices)))
        else:
            slices = list(slices)

        # Fit slicing to dimension of the dataset
        if len(slices) <= self.ndim:
            nmiss = self.ndim - len(slices)
            while Ellipsis in slices:
                idx = slices.index(Ellipsis)
                slices = slices[:idx] + ([slice(None)] * (nmiss+1)) + slices[idx+1:]
            if len(slices) < self.ndim:
                slices = list(slices) + ([slice(None)] * nmiss)
        elif len(slices) > self.ndim:
            raise DatasetException('Invalid slicing of dataset of dimension `{}`'
                                   ' with {}-dimensional slicing'
                                   .format(self.ndim, len(slices)))
        # Wrap integer slicing and `:` slicing
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
                if start < 0 or start >= self.shape[i]:
                    raise DatasetException('Only possitive and in-bounds slicing supported: `{}`'
                                           .format(slices))
                if stop < 0 or stop > self.shape[i] or stop < start:
                    raise DatasetException('Only possitive and in-bounds slicing supported: `{}`'
                                           .format(slices))
                if s.step is not None and s.step != 1:
                    raise DatasetException('Only slicing with step 1 supported')
                final_slices.append(slice(start, stop))
            else:
                raise DatasetException('Invalid type `{}` in slicing, only integer or'
                                       ' slices are supported'.format(type(s)))

        if detect_squeeze:
            return final_slices, squeeze_axis
        return final_slices

    def _get_chunk_slice_iterator(self, slices):
        indexes = []
        ltargets = []
        gtargets = []
        for slice_axis, chunk_axis_size, max_chunks in zip(slices, self.chunk_size, self.chunks):
            start_chunk = slice_axis.start // chunk_axis_size
            end_chunk = min(slice_axis.stop // chunk_axis_size, max_chunks-1)
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
                    _gslice.append(gtargets[n][j])
                yield tuple(_index), _lslice, _gslice
        return __chunk_iterator

    ###########################################################
    # DATASET SLICING
    ###########################################################

    def __getitem__(self, slices):
        slices, squeeze_axis = self._process_slices(slices, detect_squeeze=True)
        tshape = tuple(x.stop - x.start for x in slices)
        chunk_iterator = self._get_chunk_slice_iterator(slices)

        if self.njobs is None or self.njobs == 1:
            output = np.full(tshape, -1, dtype=self.dtype)
            for idx, cslice, gslice in chunk_iterator():
                output[gslice] = self._get_chunk_data(idx, slices=cslice)
        else:
            with tempfile.NamedTemporaryFile() as f:
                output = self._make_temporary_memmap(f.name, shape=tshape)
                Parallel(n_jobs=self.njobs, backend="threading")(
                    delayed(_get_chunk_data_parallel)(self, idx, cslice, gslice, output)
                    for idx, cslice, gslice in chunk_iterator()
                )
                output = np.asarray(output)

        if len(squeeze_axis) > 0:
            return np.squeeze(output, axis=squeeze_axis)  # Creates a view
        return output

    def __setitem__(self, slices, values):
        if self.read_only:
            raise DatasetException('Dataset `{}` is read-only'.format(self.name))
        slices = self._process_slices(slices)
        chunk_iterator = self._get_chunk_slice_iterator(slices)

        if self.njobs is None or self.njobs == 1:
            for idx, cslice, gslice in chunk_iterator():
                self._set_chunk_data(idx, values[gslice], slices=cslice)
        else:
            #with tempfile.NamedTemporaryFile() as f:
                #values = self._make_temporary_memmap(f.name, data=values)
            Parallel(n_jobs=self.njobs, backend="threading")(
                delayed(_set_chunk_data_parallel)(self, idx, cslice, gslice, values)
                for idx, cslice, gslice in chunk_iterator()
            )

    # Temporrary memmapping if needed
    def _make_temporary_memmap(self, filename, data=None, shape=None):
        if data is not None:
            dump(data, filename)
            return load(filename, mmap_mode='r')
        elif shape is not None:
            return np.memmap(filename, dtype=self.dtype, shape=shape, mode='w+')
        else:
            raise DatasetException('Incorrect shape or values for memmapping')

    ###########################################################
    # PROPERTIES
    ###########################################################

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
    def chunk_bytes(self):
        return self._chunk_bytes

    @property
    def total_chunks(self):
        return self._total_chunks

    @property
    def njobs(self):
        return self._njobs

    ###########################################################
    # DATASET CREATION / DELETION
    ###########################################################

    @classmethod
    def zeros(cls, pool, name, shape=None, dtype=None, **kwargs):
        return cls.create(pool, name, shape=shape, dtype=dtype,
                          data=None, fillvalue=0, **kwargs)

    @classmethod
    def ones(cls, pool, name, shape=None, dtype=None, **kwargs):
        return cls.create(pool, name, shape=shape, dtype=dtype,
                          data=None, fillvalue=1, **kwargs)

    @classmethod
    def zeros_like(cls, ds, name, **kwargs):
        return cls.create_like(ds, name, fillvalue=0, **kwargs)

    @classmethod
    def ones_like(cls, ds, name, **kwargs):
        return cls.create_like(ds, name, fillvalue=1, **kwargs)

    @classmethod
    def create_like(cls, ds, name, pool=None, njobs=None, fillvalue=-1,
                    read_only=None, chunks=None):

        pool = pool or ds.pool
        fillvalue = fillvalue or ds.fillvalue
        njobs = njobs or ds.njobs
        chunks = chunks or ds.chunk_size
        read_only = ds.read_only if read_only is None else read_only

        return cls.create(pool, name, shape=ds.shape, dtype=ds.dtype, data=None,
                          chunks=chunks, fillvalue=fillvalue, njobs=njobs,
                          read_only=read_only)

    @classmethod
    def create(cls, pool, name, shape=None, dtype=np.float32, fillvalue=-1, chunks=None,
               data=None, read_only=False, njobs=None):
        try:
            pool.stat(name)
            raise DatasetException('Object `{}` already exists in pool `{}`'.format(name, pool.name))
        except rados.ObjectNotFound:
            pass

        if data is not None:
            shape = data.shape
            dtype = data.dtype

        if shape is None:
            raise DatasetException('Invalid dataset creation, `shape` or `data` has to be specified.')

        if dtype is None:
            raise DatasetException('Invalid dataset creation, `dtype` or `data` has to be specified.')

        chunk_size = cls._validate_chunk_shape(chunks, shape)
        chunks_needed = (np.ceil(np.asarray(shape, float) / chunk_size)).astype(int)

        pool.write(name, cls.__signature__)
        pool.set_xattr(name, 'shape', shape2str(shape))
        pool.set_xattr(name, 'dtype', dtype2str(dtype))
        pool.set_xattr(name, 'fillvalue', repr(fillvalue))

        pool.set_xattr(name, 'chunks', shape2str(chunks_needed))
        pool.set_xattr(name, 'chunk_size', shape2str(chunk_size))

        ds = cls(pool, name, read_only=read_only, njobs=njobs or pool.njobs)

        if data is not None:
            ds.load(data)

        return ds

    ###########################################################
    # CHUNK NAMING AND VALIDATION
    ###########################################################

    @staticmethod
    def _validate_chunk_shape(chunks, shape):
        if chunks is None:
            return np.asarray(shape, int)
        elif type(chunks) == int:
            return np.asarray([chunks] * len(shape), int)
        elif hasattr(chunks, '__iter__') and len(chunks) == len(shape):
            return np.asarray(chunks, int)
        try:
            return np.asarray([int(chunks)] * len(shape), int)
        except ValueError:
            pass
        raise DatasetException('Dimension of chunks does not match the data shape')

    def __chunkname(self, chunk_idx):
        name = 'DataChunk.' + self.name + '.'
        name += '.'.join(map(str, chunk_idx))
        return name

    def _transform_chunk_index(self, idx):
        if type(idx) == int:
            idx = [idx]
        elif type(idx[0]) in [tuple, list] and len(idx) == 1:
            idx = idx[0]

        # Support flat indexing
        if len(idx) == 1 and self.ndim > 1 and idx < self.total_chunks:
            idx = np.unravel_index(idx[0], self.chunks)
        if len(idx) == len(self.shape):
            if any(c1 >= c2 for c1, c2 in itertools.izip(idx, self.chunks)):
                raise DatasetException('Out of limits chunk indexing of chunk `{}` with grid `{}`'
                                       .format(idx, self.chunks))
        else:
            raise DatasetException('Incorrect chunk indexing format `{}`'.format(idx))
        return idx

    ###########################################################
    # PUBLIC CHUNK MANAGEMENT
    ###########################################################

    def has_chunk(self, chunk_idx):
        chunk_idx = self._transform_chunk_index(chunk_idx)
        return self._has_chunk(chunk_idx)

    def get_chunk(self, chunk_idx):
        chunk_idx = self._transform_chunk_index(chunk_idx)
        return self._get_chunk(chunk_idx)

    def get_chunk_data(self, chunk_idx, slices=None):
        chunk_idx = self._transform_chunk_index(chunk_idx)
        if slices is not None:
            slices = self._process_slices(slices)
        return self._get_chunk_data(chunk_idx, slices=slices)

    def set_chunk_data(self, chunk_idx, data, slices=None):
        chunk_idx = self._transform_chunk_index(chunk_idx)
        if slices is not None:
            slices = self._process_slices(slices)
        self._set_chunk_data(chunk_idx, data, slices=slices)

    ###########################################################
    # PRIVATE CHUNK MANAGEMENT -- Assume processed `chunk_idx` and `slices`
    ###########################################################

    # General methods

    def _has_chunk(self, chunk_idx):
        return self.pool.has_chunk(self.__chunkname(chunk_idx))

    def _get_chunk(self, chunk_idx):
        if self._has_chunk(chunk_idx):
            return self._get_existing_chunk(chunk_idx)
        return self._create_chunk(chunk_idx)

    def _get_chunk_data(self, chunk_idx, slices=None):
        if self._has_chunk(chunk_idx):
            return self._get_existing_chunk_data(chunk_idx, slices=slices)
        elif slices is None:
            shape = self.chunk_size
        else:
            shape = [s.stop - s.start for s in slices]
        return np.full(shape, self.fillvalue, self.dtype)

    def _set_chunk_data(self, chunk_idx, data, slices=None):
        if self._has_chunk(chunk_idx):
            self._set_existing_chunk_data(chunk_idx, data, slices=slices)
        else:
            self._create_chunk(chunk_idx, data=data, slices=slices)

    # Existing chunks

    def _get_existing_chunk(self, chunk_idx):
        return DataChunk(self.pool, self.__chunkname(chunk_idx))

    def _get_existing_chunk_data(self, chunk_idx, slices=None):
        chunk = self._get_existing_chunk(chunk_idx)
        return chunk.get_data() if slices is None else chunk.get_slices(slices)

    def _set_existing_chunk_data(self, chunk_idx, data, slices=None):
        chunk = self._get_existing_chunk(chunk_idx)
        if slices is not None:
            chunk.set_slices(slices, data)
        else:
            chunk.set_data(data)

    # Create + Delete

    def _create_chunk(self, chunk_idx, data=None, slices=None):
        return DataChunk.create(self.pool, self.__chunkname(chunk_idx),
                                shape=self.chunk_size, dtype=self.dtype,
                                fillvalue=self.fillvalue, data=data, slices=slices)

    def _delete_chunk(self, chunk_idx):
        if self._has_chunk(chunk_idx):
            chunk = self._get_existing_chunk(chunk_idx)
            chunk.delete()

    ###########################################################
    # DATA MANAGEMENT
    ###########################################################

    def _gchunk_bounds_slices(self, idx):
        return [slice(i * s, min((i + 1) * s, self.shape[j]))
                for j, (i, s) in enumerate(zip(idx, self.chunk_size))]

    def _lchunk_bounds_slices(self, idx):
        return [slice(0, min((i + 1) * s, self.shape[j]) - i * s)
                for j, (i, s) in enumerate(zip(idx, self.chunk_size))]

    def load(self, data):
        chunks = self.chunks
        if self.njobs is None or self.njobs == 1:
            for idx in np.ndindex(*chunks):
                gslices = self._gchunk_bounds_slices(idx)
                lslices = self._lchunk_bounds_slices(idx)
                self._set_chunk_data(idx, data[gslices], slices=lslices)
        else:
            Parallel(n_jobs=self.njobs, backend="threading")(
                delayed(_populate_dataset)(self, idx, data)
                for idx in np.ndindex(*chunks)
            )

    def delete(self, delete_chunks=True):
        if delete_chunks:
            for idx in np.ndindex(*self.chunks):
                self._delete_chunk(idx)
        super(Dataset, self).delete()

    def clone(self, name):
        return Dataset.create(self.pool, name, shape=self.shape, dtype=self.dtype,
                              chunks=self.chunk_size, read_only=self.read_only,
                              njobs=self.njobs, fillvalue=self.fillvalue)

    ###########################################################
    # OTHER UTILITY FUNCTIONS -- Applied to every chunk
    ###########################################################

    def map(self, name, func, *args, **kwargs):
        dsout = self.clone(name)
        chunks = self.chunks

        if self.njobs is None or self.njobs == 1:
            for idx in np.ndindex(*chunks):
                slices = self._lchunk_bounds_slices(idx)
                result = func(self._get_chunk_data(idx, slices=slices), *args, **kwargs)
                dsout._set_chunk_data(idx, result, slices=slices)
        else:
            Parallel(n_jobs=self.njobs, backend="threading")(
                delayed(_map_parallel)(self, dsout, func, idx, *args, **kwargs)
                for idx in np.ndindex(*chunks)
            )
        return dsout

    def apply(self, func, *args, **kwargs):
        chunks = self.chunks
        if self.njobs is None or self.njobs == 1:
            for idx in np.ndindex(*chunks):
                slices = self._lchunk_bounds_slices(idx)
                result = func(self._get_chunk_data(idx, slices=slices), *args, **kwargs)
                self._set_chunk_data(idx, result, slices=slices)
        else:
            Parallel(n_jobs=self.njobs, backend="threading")(
                delayed(_apply_parallel)(self, func, idx, *args, **kwargs)
                for idx in np.ndindex(*chunks)
            )


# Parallel bindings

def _populate_dataset(inst, idx, data):
    gslices = inst._gchunk_bounds_slices(idx)
    lslices = inst._lchunk_bounds_slices(idx)
    inst._set_chunk_data(idx, data[gslices], slices=lslices)


def _get_chunk_data_parallel(inst, chunk_idx, cslice, gslice, doutput):
    doutput[gslice] = inst._get_chunk_data(chunk_idx, slices=cslice)


def _set_chunk_data_parallel(inst, chunk_idx, cslice, gslice, dinput):
    inst._set_chunk_data(chunk_idx, dinput[gslice], slices=cslice)


def _map_parallel(inst_in, inst_out, func, chunk_idx, *args, **kwargs):
    slices = inst_in._lchunk_bounds_slices(chunk_idx)
    result = func(inst_in._get_chunk_data(chunk_idx, slices=slices), *args, **kwargs)
    inst_out._set_chunk_data(chunk_idx, result, slices=slices)


def _apply_parallel(inst, func, chunk_idx, *args, **kwargs):
    slices = inst._lchunk_bounds_slices(chunk_idx)
    result = func(inst._get_chunk_data(chunk_idx, slices=slices), *args, **kwargs)
    inst._set_chunk_data(chunk_idx, result, slices=slices)