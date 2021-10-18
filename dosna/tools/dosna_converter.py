#!/usr/bin/env python
""" Tool to translate HDF5 files to DosNa"""

import numpy as np

import h5py
import json
from contextlib import contextmanager

_SHAPE = 'shape'
_DTYPE = 'dtype'
_NDIM = 'ndim'
_NBYTES = 'nbytes'
_FILLVALUE = 'fillvalue'
_CHUNK_SIZE = 'chunk_size'
_CHUNK_GRID = 'chunk_grid'
_IS_DATASET = 'is_dataset'
_DATASET_NAME = 'name'
_DATASET_VALUE = 'dataset_value'
_PATH_SPLIT = '/'
_ABSOLUTE_PATH = 'absolute_path'

_METADATA = 'metadata'
_ATTRS = 'attrs'
_LINKS = 'links'

@contextmanager
def hdf_file(hdf, *args, **kwargs):
    if isinstance(hdf, str):
        yield h5py.File(hdf, 'r',*args, **kwargs)
    else:
        yield hdf

class Dosnatohdf5(object):

    def __init__(self, connection):
        self._connection = connection

    def dosna2dict(self):

        def _recurse(group, links, dosnadict):
            for key, value in links.items():
                object = value.target
                if hasattr(object, _LINKS):
                    subgroup = group.get_group(key)
                    dosnadict[key] = dict()
                    dosnadict[key][_ATTRS] = subgroup.attrs
                    dosnadict[key] = _recurse(subgroup, subgroup.links, dosnadict[key])
                else:
                    dosnadict[key] = group.get_dataset(key)
            return dosnadict

        dosnadict = _recurse(self._connection, self._connection.root_group.links, {})

        return dosnadict

    def dosna2hdf(self, h5file):
        dosnadict = self.dosna2dict()

        def _recurse(dosnadict, hdfobject):
            for key, value in dosnadict.items():
                if isinstance(value, dict):
                    if key != _ATTRS:
                        if not key in list(hdfobject.keys()):
                            hdfgroup = hdfobject.create_group(key)
                            for k, v in value[_ATTRS].items():
                                hdfgroup.attrs[k] = v
                            _recurse(value, hdfgroup)
                        else:
                            raise Exception('Group: ', key, 'already created.')
                else:
                    if not key in list(hdfobject.keys()):
                        dataset = hdfobject.create_dataset(
                            key,
                            shape=value.shape,
                            chunks=value.chunk_size,
                            dtype=value.dtype,
                        )
                        if dataset.chunks is not None:
                            for s in dataset.iter_chunks():
                                dataset[s] = value[s]
                    else:
                        raise Exception('Dataset', key, 'already created.')

        with h5py.File(h5file, 'w') as hdf:
            _recurse(dosnadict, hdf)
            return hdf

    def dosna2json(self, jsonfile):
        dosnadict = self.dosna2dict()
        def _recurse(dosnadict, jsondict):
            for key, value in dosnadict.items():
                if isinstance(value, dict):
                    if key != _ATTRS:
                        jsondict[key] = dict()
                        jsondict[key][_ATTRS] = value[_ATTRS]
                        jsondict[key] = _recurse(value, jsondict[key])
                else:
                    jsondict[key] = dict()
                    jsondict[key][_DATASET_NAME] = key
                    jsondict[key][_ABSOLUTE_PATH] = value.get_absolute_path()
                    jsondict[key][_SHAPE] = value.shape
                    jsondict[key][_NDIM] = value.ndim
                    jsondict[key][_DTYPE] = value.dtype.str
                    jsondict[key][_FILLVALUE] = float(value.fillvalue)
                    jsondict[key][_CHUNK_SIZE] = value.chunk_size
                    jsondict[key][_CHUNK_GRID] = value.chunk_grid.tolist()
                    jsondict[key][_IS_DATASET] = True
                    data = value[:]
                    jsondict[key][_DATASET_VALUE] = data.tolist()
            return jsondict

        jsondict = _recurse(dosnadict, {})

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

    # def json_to_hdf5(self, jsonfile, h5file):
    #
    #     with open(jsonfile, 'r') as f:
    #         jsondict = json.loads(f.read())
    #
    #     def _recurse(jsondict, hdf5dict, group):
    #         for key, value in jsondict.items():
    #             if isinstance(value, dict):
    #                 if not _IS_DATASET in value:
    #                     subgroup = group.get(key)
    #                     _recurse(value, hdf5dict, subgroup)
    #                 else:
    #                     dataset = group.get(key)
    #
    #     with h5py.File(h5file, "r") as hdf:
    #         _recurse(jsondict, {}, hdf)
    #         return hdf
    #
    # def hdf5file_to_hdf5dict(self, hdf5file):
    #     def load(hdf):
    #         def _recurse(hdfobject, datadict):
    #             for key, value in hdfobject.items():
    #                 if isinstance(value, h5py.Group):
    #                     datadict[key] = dict()
    #                     attrs = dict()
    #                     for k, v in value.attrs.items():
    #                         attrs[k] = v
    #                     datadict[key][_ATTRS] = attrs
    #                     datadict[key] = _recurse(value, datadict[key])
    #                 elif isinstance(value, h5py.Dataset):
    #                     datadict[key] = value
    #             return datadict
    #
    #         with hdf_file(hdf) as hdf:
    #             data = dict(_h5file=hdf)
    #             return _recurse(hdf, data)
    #
    #     hdf5dict = load(hdf5file)
    #     return hdf5dict