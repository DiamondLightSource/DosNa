#!/usr/bin/env python
""" Tool to translate HDF5 files to DosNa"""

import logging

import numpy as np

import uuid

import h5py
import json
import dosna as dn
import dosna.tools.hdf5todict as hd
from dosna.tools.hdf5todict import LazyHdfDict #TODO: Create own methods?
from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset, DatasetNotFoundError,
                                 BackendGroup, GroupNotFoundError)

from h5py._hl.dataset import Dataset

class Hdf5todosna():
    """
    Includes methods to transform HDF5 to DosNa Objects
    """

    def __init__(self, h5file=None, max_num_bytes=5, *args, **kwargs):
        self._h5file = h5file
        self.max_num_bytes = max_num_bytes

    def hdf5_to_dict(self):
        hdf5dict = hd.load(self._h5file, lazy=True)
        return hdf5dict
    
    def hdf5dict_to_dosna(self, hdf5dict, dosnaconnection):
        
        def _create_dataset(name, h5_dataset, group):
            if h5_dataset.nbytes > self.max_num_bytes:
                dosna_dataset = group.create_dataset(
                                name,
                                shape=h5_dataset.shape,
                                dtype=h5_dataset.dtype,
                                chunk_size=h5_dataset.chunks)
                if h5_dataset.chunks is not None:
                    for chunk in h5_dataset.iter_chunks():
                        dosna_dataset[chunk] = h5_dataset[chunk]
                else:
                    pass # TODO copy dataset?
            else:
                dosna_dataset = np.zeros(h5_dataset.shape) # TODO: this doesn't get stored
                h5_dataset.read_direct(dosna_dataset)
            return dosna_dataset
        
        def _recurse(hdf5dict, dosnadict, group):
            for key, value in hdf5dict.items():
                if isinstance(value, LazyHdfDict):
                    dosnadict[key] = {}
                    subgroup = group.create_group(key, value["metadata"])
                    dosnadict[key] = _recurse(value, dosnadict[key], subgroup)
                else:
                    if isinstance(value, Dataset):
                        dosna_dataset = _create_dataset(key, value, group)
                        dosnadict[key] = dosna_dataset 
            return dosnadict
            

        return _recurse(hdf5dict, {}, dosnaconnection)


    def hdf5dict_to_json(self, hdf5dict, jsonfile): #TODO specify which type of json you are returning
        
        def _recurse(hdf5dict, jsondict):
            for key, value in hdf5dict.items():
                if isinstance(value, LazyHdfDict):
                    jsondict[key] = {}
                    jsondict[key] = _recurse(value, jsondict[key])
                elif isinstance(value, Dataset):
                    jsondict[key] = {}
                    jsondict[key]["name"] = key # TODO path = key.split("/")
                    jsondict[key]["shape"] = value.shape
                    jsondict[key]["ndim"] = value.ndim
                    jsondict[key]["dtype"] = value.dtype.str 
                    jsondict[key]["nbytes"] = value.nbytes.item()
                    jsondict[key]["is_dataset"] = True
                    jsondict[key]["absolute_path"] = value.name
                    if value.nbytes < self.max_num_bytes:
                        data_np = np.zeros(value.shape)
                        value.read_direct(data_np)
                        jsondict[key]["value"] = data_np.tolist()
                        
            return jsondict
        
        with open(jsonfile, 'w') as f:
            f.write(json.dumps(jsondict))
        
        return _recurse(hdf5dict, {})

    
    def json_to_dosna(self, jsonfile, dosnaobject):
        
        with open(jsonfile, 'r') as f:
            jsonstring = f.read()
        jsondict = json.loads(jsonstring)
        
        def _recurse(jsondict, dosnadict, group):
            for key, value in jsondict.items():
                if isinstance(value, dict):
                    if "is_dataset" in value:
                        dataset = group.get_dataset(key)
                        dosnadict[key] = dataset
                    else:
                        subgroup = group.get_group(key)
                        dosnadict[key] = {}
                        dosnadict[key] = _recurse(value, dosnadict[key], subgroup)
            return dosnadict
        
        return _recurse(jsondict, {}, dosnaobject)