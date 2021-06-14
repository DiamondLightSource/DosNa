#!/usr/bin/env python
""" Tool to translate HDF5 files to DosNa"""

import logging

import time
import uuid
import h5py
import json
import numpy as np

import dosna as dn
from dosna.tools.hdf5todict import LazyHdfDict  # TODO: Create own methods?
from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset, DatasetNotFoundError,
                                 BackendGroup, GroupNotFoundError,
)

import multiprocessing
from contextlib import contextmanager


# TODO leave it, is copy?
@contextmanager
def hdf_file(hdf, *args, **kwargs):
    """Context manager yields h5 file if hdf is str,
    otherwise just yield hdf as is."""
    if isinstance(hdf, str):
        yield h5py.File(hdf, 'r',*args, **kwargs)
    else:
        yield hdf

def bytes_to_mb(size_bytes):
    if size_bytes == 0:
        return "0B"
    return round((size_bytes / 1024 / 1024), 2)

_SHAPE = 'shape'
_DTYPE = 'dtype'
_NDIM = 'ndim'
_NBYTES = 'nbytes'
_CHUNK_SIZE = 'chunk_size'
_IS_DATASET = 'is_dataset'
_DATASET_NAME = 'name'
_DATASET_VALUE = 'dataset_value'
_PATH_SPLIT = '/'
_ABSOLUTE_PATH = 'absolute_path'

_METADATA = 'metadata' # TODO maybe change this
_ATTRS = 'attrs'

class Hdf5todosna(object):

    def __init__(self, h5file=None, max_num_mb=100, *args, **kwargs):
        self._h5file = h5file
        self._size = max_num_mb # TODO change name

    def hdf5_to_dict(self):
        def load(hdf):
            def _recurse(hdfobject, datadict):
                for key, value in hdfobject.items():
                    if isinstance(value, h5py.Group):
                        datadict[key] = LazyHdfDict() # TODO lazy HDf5fidct
                        attrs = dict()
                        for k, v in value.attrs.items():
                            attrs[k] = v
                        datadict[key][_ATTRS] = attrs
                        datadict[key] = _recurse(value, datadict[key])
                    elif isinstance(value, h5py.Dataset):
                        # if new:
                        #    key = key + "-" + str(uuid.uuid4()) # TODO
                        datadict[key] = value
                return datadict

            with hdf_file(hdf) as hdf:
                data = LazyHdfDict(_h5file=hdf)
                return _recurse(hdf, data)

        hdf5dict = load(self._h5file)
        return hdf5dict

    def hdf5dict_to_dosna(self, hdf5dict, dosnaconnection):

        def _create_dataset(name, h5_dataset, group):
            #print("DATASET SIZE", convert_size(h5_dataset.nbytes), h5_dataset.name) # TODO remove print
            if bytes_to_mb(h5_dataset.nbytes) < self._size:
                data = np.zeros(h5_dataset.shape, dtype=h5_dataset.dtype)
                h5_dataset.read_direct(data)
                dosna_dataset = group.create_dataset(
                    name,
                    data = data,
                )
            else:
                dosna_dataset = group.create_dataset(
                    name,
                    shape= h5_dataset.shape,
                    dtype=h5_dataset.dtype,
                    chunk_size=h5_dataset.chunks,
                )
                if h5_dataset.chunks is not None:
                    for chunk in h5_dataset.iter_chunks():
                        dosna_dataset[chunk] = h5_dataset[chunk]
                else:
                    raise Exception('Dataset size bigger than`{}` MB and is not chunked'.format(self._size))

            return dosna_dataset

        def _recurse(hdf5dict, dosnadict, group):
            for key, value in hdf5dict.items():
                if isinstance(value, LazyHdfDict):
                    subgroup = group.create_group(key, value[_ATTRS])
                    dosnadict[key] = dict()
                    dosnadict[key][_ATTRS] = value[_ATTRS]
                    dosnadict[key] = _recurse(value, dosnadict[key], subgroup)
                else:
                    if isinstance(value, h5py.Dataset):
                        dosna_dataset = _create_dataset(key, value, group)
                        dosnadict[key] = dosna_dataset
            return dosnadict

        dosnadict = _recurse(hdf5dict, {}, dosnaconnection)
        return dosnadict

    def hdf5dict_to_json(self, hdf5dict, jsonfile):
        def _recurse(hdf5dict, jsondict):
            for key, value in hdf5dict.items():
                if isinstance(value, LazyHdfDict): # TODO change this
                    jsondict[key] = dict()
                    jsondict[key][_ATTRS] = value[_ATTRS]
                    jsondict[key] = _recurse(value, jsondict[key])
                elif isinstance(value, h5py.Dataset):
                    jsondict[key] = dict()
                    jsondict[key][_DATASET_NAME] = key  # TODO path = key.split("/") # TODO but if I am changing this, change everything
                    jsondict[key][_SHAPE] = value.shape
                    jsondict[key][_CHUNK_SIZE] = value.chunks
                    jsondict[key][_NDIM] = value.ndim
                    jsondict[key][_DTYPE] = value.dtype.str
                    jsondict[key][_NBYTES] = value.nbytes.item()
                    jsondict[key][_IS_DATASET] = True
                    jsondict[key][_ABSOLUTE_PATH] = value.name
                    if bytes_to_mb(value.nbytes) < self._size:
                        data = np.zeros(value.shape, value.dtype)
                        value.read_direct(data)
                        jsondict[key][_DATASET_VALUE] = data.tolist()

            return jsondict

        jsondict = _recurse(hdf5dict, {})

        def json_encoder(obj):
            if isinstance(obj, np.ndarray):
                object_list = obj.tolist()
                return [str(x) for x in object_list]
            if isinstance(obj, bytes):
                return str(obj)
            raise TypeError('Not serializable: ', type(obj))

        with open(jsonfile, 'w') as f:
            f.write(json.dumps(jsondict, default=json_encoder))

        return jsondict

    def json_to_dosna(self, jsonfile, dosnaobject):

        with open(jsonfile, 'r') as f:
            jsonstring = f.read()
        jsondict = json.loads(jsonstring)

        def _recurse(jsondict, dosnadict, group):
            for key, value in jsondict.items():
                if not _IS_DATASET in value:
                    if key != _ATTRS:
                        subgroup = group.get_group(key)
                        dosnadict[key] = dict()
                        dosnadict[key][_ATTRS] = value[_ATTRS]
                        dosnadict[key] = _recurse(value, dosnadict[key], subgroup)
                else:
                    dataset = group.get_dataset(key)
                    dosnadict[key] = dataset

            return dosnadict

        dosnadict = _recurse(jsondict, {}, dosnaobject)
        return dosnadict