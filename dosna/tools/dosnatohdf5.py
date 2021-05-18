#!/usr/bin/env python
""" Tool to translate HDF5 files to DosNa"""

import logging

import numpy as np

import os
import h5py
import json
import dosna as dn
import hdf5todict.hdf5todict as hd
from hdf5todict.hdf5todict import LazyHdfDict
from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset, DatasetNotFoundError)

class Dosnatohdf5():
    """
    Includes methods to transform HDF5 to DosNa Objects
    """

    def __init__(self, dnconnection=None):
        self._dnconnection = dnconnection
        
    def dosna_to_dict(self):
        pass
    
    def dosnadict_to_hdf5(self, dosnadict, h5file):
        f = h5py.File(h5file, "w") #TODO remove this
        
        def _recurse(dosnadict, hdfobject):
            for key, value in dosnadict.items():
                if isinstance(value, dict):
                    if not key in list(hdfobject.keys()): #TODO this includes groups and datasets
                        hdfgroup = hdfobject.create_group(key)
                        _recurse(value, hdfgroup)
                    else:
                        raise Exception("Group already created")

                elif isinstance(value, dn.engines.cpu.CpuDataset): # TODO type dosna.engines.cpu.CpuDataset
                    if not key in list(hdfobject.keys()): #TODO includes groups and datasets
                        dataset = hdfobject.create_dataset(
                            key,
                            shape=value.shape,
                            chunks=value.chunk_size,
                            dtype=value.dtype
                        )
                        if dataset.chunks is not None:
                            for s in dataset.iter_chunks():
                                dataset[s] = value[s]
            
        with h5py.File(h5file) as hdf:
            _recurse(dosnadict, hdf)
            return hdf
        
    def dosnadict_to_jsondict(self, dosnadict):
        
        def _recurse(dosnadict, jsondict):
            for key, value in dosnadict.items():
                #jsondict[key] = {} #TODO: mira esto?
                if isinstance(value, dict):
                    jsondict[key] = {}
                    jsondict[key] = _recurse(value, jsondict[key])
                elif isinstance(value, dn.engines.cpu.CpuDataset):
                    jsondict[key] = value.name
            return jsondict
        
        return _recurse(dosnadict, {})
    
    def jsondict_to_jsonfile(self, jsondict, jsonfile):
        with open(jsonfile, 'w') as f:
            f.write(json.dumps(jsondict))
        return jsonfile

    def jsonfile_to_jsondict(self, jsonfile):
        with open(jsonfile, 'r') as f:
            jsondict = json.loads(f.read())
        return jsondict

    def jsondict_to_hdf5(self, jsondict, h5file):
        #is it the hdf5dict or is it the hdf5file
        h = h5py.File(h5file, "r")
        def _recurse(jsondict, hdfobject):
            for key, value in jsondict.items():
                if isinstance(value, dict):
                    hdfgroup = hdfobject.get(key)
                    print(key)
                    print(hdfgroup)
                    _recurse(value, hdfobject)
                    """
                elif isinstance(value, dn.engines.cpu.CpuDataset):
                    print("KEY", key)
                    print("VALUE", value)
                     # TODO type dosna.engines.cpu.CpuDataset
                    if not key in list(hdfobject.keys()): #TODO includes groups and datasets
                        dataset = hdfobject.create_dataset(
                            key,
                            shape=value.shape,
                            chunks=value.chunk_size,
                            dtype=value.dtype
                        )
                        if dataset.chunks is not None:
                            for s in dataset.iter_chunks():
                                dataset[s] = value[s]
                    """
            
        with h5py.File(h5file) as hdf:
            _recurse(jsondict, hdf)
            return hdf
