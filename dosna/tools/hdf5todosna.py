#!/usr/bin/env python
""" Tool to translate HDF5 files to DosNa"""

import logging

import numpy as np

import h5py
import json
import dosna as dn
import hdf5todict.hdf5todict as hd
from hdf5todict.hdf5todict import LazyHdfDict #TODO: Create own methods?
from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset, DatasetNotFoundError,
                                 BackendGroup, GroupNotFoundError)

# TODO: do I have to inherit from Object or create a superclass?

class Hdf5todosna():
    """
    Includes methods to transform HDF5 to DosNa Objects
    """

    def __init__(self, h5file=None, *args, **kwargs):
        self._h5file = h5file

    def hdf5_to_dict(self):
        # TODO: here is the hdf5file or in the init method?
        # TODO: AttributeError: 'NoneType' object has no attribute 'items'
        # TODO: Why not unite these two methods into one?
        hdf5dict = hd.load(self._h5file, lazy=True)
        return hdf5dict
    
    def hdf5dict_to_dosna(self, datadict, dosnaobject):
        
        # TODO: what happens to dosnadict?
        
        def _recurse(datadict, dosnadict):
            # TODO: mirar la class del dictionary
            for key, value in datadict.items():
                if isinstance(value, LazyHdfDict):
                    dosnadict[key] = {}
                    dosnadict[key][key] = dosnaobject.create_tree(key)
                    dosnadict[key] = _recurse(value, dosnadict[key])
                elif isinstance(value, h5py._hl.dataset.Dataset):
                    dataset = dosnaobject.create_dataset(
                        key,
                        shape=value.shape,
                        dtype=value.dtype,
                        chunk_size=value.chunks,
                    )
                    if value.chunks is not None:
                        for s in value.iter_chunks():
                            dataset[s] = value[s]
                    else:
                        
                    dosnadict[key] = dataset
            return dosnadict
        return _recurse(datadict, {})
        

    def hdf5dict_to_json(self, hdf5dict):
        
        def _recurse(hdf5dict, jsondict):
            for key, value in hdf5dict.items():
                jsondict[key] = {}
                if isinstance(value, LazyHdfDict):
                    jsondict[key] = _recurse(value, jsondict[key])
                elif isinstance(value, h5py.Dataset):
                    #jsondict[key]["name"] = value.name
                    #jsondict[key]["shape"] = value.shape
                    jsondict[key] = value.name
            return jsondict
        
        jsondict =_recurse(hdf5dict, {})
        jsonstring = json.dumps(jsondict)
        
        with open('file_to_write.json', 'w') as f:
            f.write(jsonstring)
        
        return jsonstring
    
    def json_to_dict(self, jsonfile):
        
        with open(jsonfile, 'r') as f:
            jsonstring = f.read()
        jsondict = json.loads(jsonstring)
        return jsondict

    def dict_to_dosna(self, datadict, dosnaobject):

        def _recurse(datadict, dosnadict):
            for key, value in datadict.items():
                if isinstance(value, dict):
                    dosnadict[key] = {}
                    dosnadict[key][key] = dosnaobject.create_tree(key)
                    dosnadict[key] = _recurse(value, dosnadict[key])
                elif isinstance(value, str):
                    dataset = dosnaobject.create_dataset(value,
                                                         shape=(100,100,100)) # TODO: change
                    dosnadict[key] = dataset
                elif isinstance(value, h5py._hl.dataset.Dataset):
                    dataset = dosnaobject.create_dataset(
                        key,
                        shape=value.shape,
                        dtype=value.dtype,
                        chunk_size=value.chunks,
                    )
                    print(dataset)
                    if value.chunks is not None:
                        for s in value.iter_chunks():
                            dataset[s] = value[s]
                            print(dataset[s])
                    dosnadict[key] = dataset
            return dosnadict
        return _recurse(datadict, {})




x = Hdf5todosna('testfile.h5')
hdf5dict = x.hdf5_to_dict()

con = dn.Connection("dosna")
con.connect()
dosnadict = x.hdf5dict_to_dosna(hdf5dict, con)
print((dosnadict))

