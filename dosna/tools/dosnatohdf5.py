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

    def __init__(self, connection=None):
        self._connection = connection
        self.max_num_bytes = 5
        
    def dosna_to_dict(self):
        links = self._connection.root_group.links

        def _recurse(dosnadict, links, group):
            for key, value in links.items():
                subgroup = group.get_group(key)
                if hasattr(subgroup, "links"):
                    sub_links = subgroup.links
                    dosnadict[key] = {}
                    dosnadict[key] = _recurse(dosnadict[key], sub_links, subgroup)
                else:
                    dosnadict[key] = group.get_dataset(key)
            return dosnadict
        dosnadict = _recurse({}, links, self._connection)
        return dosnadict
    
    def dosnadict_to_hdf5(self, dosnadict, h5file):
        
        def _recurse(dosnadict, hdfobject):
            for key, value in dosnadict.items():
                if isinstance(value, dict):
                    if not key in list(hdfobject.keys()):
                        hdfgroup = hdfobject.create_group(key)
                        _recurse(value, hdfgroup)
                    else:
                        raise Exception("Group", key, "already created")
                else:
                    if not key in list(hdfobject.keys()):
                        dataset = hdfobject.create_dataset(
                            key,
                            shape=value.shape,
                            chunks=value.chunk_size,
                            dtype=value.dtype
                        )
                    else:
                        raise Exception("Dataset", key, "already created")
                        if dataset.chunks is not None:
                            for s in dataset.iter_chunks():
                                dataset[s] = value[s]
        with h5py.File(h5file, "w") as hdf:
            _recurse(dosnadict, hdf)
            return hdf
        
    def dosnadict_to_jsondict(self, dosnadict, jsonfile):
        
        def _recurse(dosnadict, jsondict):
            for key, value in dosnadict.items():
                if isinstance(value, dict):
                    jsondict[key] = {}
                    jsondict[key] = _recurse(value, jsondict[key])
                else:
                    jsondict[key] = {}
                    jsondict[key]["name"] = key # TODO path = key.split("/")
                    jsondict[key]["shape"] = value.shape
                    jsondict[key]["dtype"] = str(value.dtype)
                    jsondict[key]["fillvalue"] = value.fillvalue
                    jsondict[key]["chunk_size"] = value.chunk_size
                    jsondict[key]["chunk_grid"] = value.chunk_grid.tolist()
                    jsondict[key]["is_dataset"] = True
                    #jsondict[key]["absolute_path"] = value.name # TODO absolute path
            return jsondict
        
        jsondict =  _recurse(dosnadict, {})
        
        with open(jsonfile, 'w') as f:
            f.write(json.dumps(jsondict))
        
        return jsondict

    def json_to_hdf5(self, jsonfile, h5file):
        
        with open(jsonfile, 'r') as f:
            jsondict = json.loads(f.read())
        
        def _recurse(jsondict, hdf5dict, group):
            for key, value in jsondict.items():
                if isinstance(value, dict):
                    if "is_dataset" in value:
                        dataset = group.get(key)
                    else:
                        subgroup = group.get(key)
                        _recurse(value, hdf5dict, subgroup)
        with h5py.File(h5file, "r") as hdf:
            _recurse(jsondict, {}, hdf)
            return hdf
        
    def hdf5file_to_hdf5dict(self, hdf5file):
        hdf5dict = hd.load(hdf5file)
        return hdf5dict
       
"""
con = dn.Connection("dn-connection") 
con.connect()
metadata = {"a1": "value"}
A = con.create_group("A", metadata)
B = A.create_group("B", metadata)
C = A.create_group("C", metadata)
D = B.create_group("D", metadata)

dset1 = B.create_dataset("dset1", shape=(3,3))
dset2 = B.create_dataset("dset2", shape=(3,3))
dset3 = B.create_dataset("dset3", shape=(3,3))

x = Dosnatohdf5(con)
dosnadict = x.dosna_to_dict()
print(dosnadict)
hdf5 = x.dosnadict_to_hdf5(dosnadict, "test_file.h5")
jsondict = x.dosnadict_to_jsondict(dosnadict, "dosna_file.json")
hdf52 = x.json_to_hdf5("dosna_file.json", "test_file.h5")
hdfdict = x.hdf5file_to_hdf5dict("test_file.h5")
print(hdfdict)
#print(jsondict)

with h5py.File("test_file.h5", 'r') as f:
    print(f.keys())
    print(f["A"].keys())
    print(f["A/B"].keys())
    

with h5py.File("test_file_2.h5", 'r') as f:
    print(f.keys())
    print(f["A"].keys())
    print(f["A/B"].keys())
"""