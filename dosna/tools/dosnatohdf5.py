#!/usr/bin/env python
""" Tool to translate HDF5 files to DosNa"""

import logging

import numpy as np

import os
import h5py
import json
import dosna as dn
import hdf5todict as hd
from hdf5todict import LazyHdfDict
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
        root_group = self._dnconnection.root_group
        def _recurse(links, dosnadict):
            for key, value in links.items():
                dosnadict[key] = {}
                if hasattr(value.target, "links"):
                    links = value.target.links
                    dosnadict[key] = _recurse(links, dosnadict[key])
                if hasattr(value.target, "shape"):
                    dataset = value.target
                    dosnadict[key] = dataset
            return dosnadict
        dosnadict = _recurse(root_group.links, {})
        return dosnadict 
    
    def dosnadict_to_hdf5(self, dosnadict, h5file):
        
        def _recurse(dosnadict, hdfobject):
            for key, value in dosnadict.items():
                if isinstance(value, dict):
                    if not key in list(hdfobject.keys()):
                        hdfgroup = hdfobject.create_group(key)
                        _recurse(value, hdfgroup)
                    else:
                        raise Exception("Group already created")

                else:
                    if not key in list(hdfobject.keys()):
                        dataset = hdfobject.create_dataset(
                            key,
                            shape=value.shape,
                            chunks=value.chunk_size,
                            dtype=value.dtype
                        )
                        """
                        if dataset.chunks is not None:
                            for s in dataset.iter_chunks():
                                dataset[s] = value[s]
                        """
            
        with h5py.File(h5file, "w") as hdf:
            _recurse(dosnadict, hdf)
            return hdf
        
    def dosnadict_to_jsondict(self, dosnadict):
        
        def _recurse(dosnadict, jsondict):
            for key, value in dosnadict.items():
                if isinstance(value, dict):
                    jsondict[key] = {}
                    jsondict[key] = _recurse(value, jsondict[key])
                else:
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
        
        def _recurse(jsondict, hdfobject):
            for key, value in jsondict.items():

                if isinstance(value, dict):
                    hdfgroup = hdfobject.get(key)
                    _recurse(value, hdfgroup)
                else:
                    hdfdataset = hdfobject.get(key)

        with h5py.File(h5file, "r") as hdf:
            _recurse(jsondict, hdf)
            return hdf

