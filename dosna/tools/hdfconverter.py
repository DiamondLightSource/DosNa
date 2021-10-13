#!/usr/bin/env python
""" Tool to translate HDF5 files to DosNa"""

import logging

import time
import uuid
import h5py
import json
import numpy as np
import multiprocessing
from contextlib import contextmanager

import dosna as dn
from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset, DatasetNotFoundError,
                                 BackendGroup, GroupNotFoundError)

@contextmanager
def read_hdf_file(hdf, *args, **kwargs):
    if isinstance(hdf, str):
        yield h5py.File(hdf, 'r',*args, **kwargs)
    else:
        yield hdf

def bytes_to_mb(size_bytes):
    if size_bytes == 0:
        return 0.0
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
_FILLVALUE = 'fillvalue'
_METADATA = 'metadata'
_ATTRS = 'attrs'

class HdfConverter(object):

    def __init__(self, max_num_mb=100, *args, **kwargs):
        self._size = max_num_mb

    def hdf2dict(self, hdf_file):
        def load(hdf):
            root_group_lists = []
            def _recurse(hdfobject, datadict):
                for object in hdfobject.values():
                    if type(object) == h5py.Group:
                        root_group_lists.append(object.name)
                for key, value in hdfobject.items():
                    if isinstance(value, h5py.Group):
                        datadict[key] = dict()
                        attrs = dict()
                        for k, v in value.attrs.items():
                            attrs[k] = v
                        datadict[key][_ATTRS] = attrs
                        datadict[key] = _recurse(value, datadict[key])
                    elif isinstance(value, h5py.Dataset):
                        datadict[key] = value
                return datadict

            with read_hdf_file(hdf) as hdf:
                data = dict(_h5file=hdf)
                return _recurse(hdf, data)

        hdfdict = load(hdf_file)
        return hdfdict

    def hdf2dosna(self, hdf_file, dn_connection):
        hdf_dict = self.hdf2dict(hdf_file)

        def _create_dataset(name, h5_dataset, group):
            if h5_dataset.chunks is None:
                if bytes_to_mb(h5_dataset.nbytes) < self._size:
                    data = np.zeros(h5_dataset.shape, dtype=h5_dataset.dtype)
                    h5_dataset.read_direct(data)
                    dosna_dataset = group.create_dataset(
                        name,
                        data=data,
                    )
                else:
                    raise Exception('Dataset size bigger than`{}` MB and is not chunked'.format(self._size))
            else:
                dosna_dataset = group.create_dataset(
                    name,
                    shape=h5_dataset.shape,
                    dtype=h5_dataset.dtype,
                    chunk_size=h5_dataset.chunks,
                )
                for chunk in h5_dataset.iter_chunks():
                    dosna_dataset[chunk] = h5_dataset[chunk]

            return dosna_dataset

        def _recurse(hdf5dict, group):
            for key, value in hdf5dict.items():
                if isinstance(value, dict) and key != _ATTRS:
                    subgroup = group.create_group(key, value[_ATTRS])
                    _recurse(value, subgroup)
                else:
                    if isinstance(value, h5py.Dataset):
                        _create_dataset(key, value, group)
            return group

        dn_connection = _recurse(hdf_dict, dn_connection)
        return dn_connection

    def hdf2json(self, hdf_file, jsonfile):
        hdfdict = self.hdf2dict(hdf_file)

        def _recurse(hdf5dict, jsondict):
            for key, value in hdf5dict.items():
                if isinstance(value, dict) and key != _ATTRS:
                    jsondict[key] = dict()
                    jsondict[key][_ATTRS] = value[_ATTRS]
                    jsondict[key] = _recurse(value, jsondict[key])
                elif isinstance(value, h5py.Dataset):
                    jsondict[key] = dict()
                    jsondict[key][_DATASET_NAME] = key
                    jsondict[key][_SHAPE] = value.shape
                    jsondict[key][_CHUNK_SIZE] = value.chunks
                    jsondict[key][_NDIM] = value.ndim
                    jsondict[key][_DTYPE] = value.dtype.str
                    jsondict[key][_NBYTES] = value.nbytes.item()
                    jsondict[key][_IS_DATASET] = True
                    jsondict[key][_ABSOLUTE_PATH] = value.name
                    jsondict[key][_FILLVALUE] = float(value.fillvalue)
                    data = np.zeros(value.shape, value.dtype)
                    value.read_direct(data)
                    jsondict[key][_DATASET_VALUE] = data.tolist()
            return jsondict

        jsondict = _recurse(hdfdict, {})

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
