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

class dosnatohdf5():
    """
    Includes methods to transform HDF5 to DosNa Objects
    """

    def __init__(self, dosnaobject):
        self.file = dosnaobject

    def dosna_to_json(self):
        json_dict = {}
        for key, value in self.file.datasets.items():
            json_dict["dataset"] = key #unique name?
        json_string = json.dumps(json_dict)
        with open('file_to_write.json', 'w') as f:
            f.write(json_string)

    def json_to_dict(self, jsonfile):
        with open(jsonfile, 'r') as f:
            json_string = f.read()
        json_dict = json.loads(json_string)
        return json_dict

    def dict_to_hdf5(self, jsonfile, dosnatree, dosnaconnection):
        h = h5py.File("newtest.h5", "w")
        def recurse(dosnadict, h5file):
            for key, value in dosnadict.items():
                if isinstance(value, dict):
                    if not key in list(h5file.keys()):
                        group = h5file.create_group(key)
                    else:
                        raise Exception("Group already created")
                elif isinstance(value, dn.backends.ram.MemDataset):
                    if not key in list(h5file.keys()):
                        dataset = h5file.create_dataset(
                            key,
                            shape=value.shape,
                            chunks=value.chunk_size,
                            dtype=value.dtype
                        )
                        #if dataset.chunks is not None:
                        #    for s in dataset.iter_chunks():
                        #        print(value[s])
                    else:
                        raise Exception("Dataset already created")
                elif isinstance(value, str):
                    cn = dn.Connection("dosna")
                    cn.connect()
                    first_tree = cn.create_tree("first_tree")
                    #dndataset = first_tree.create_dataset("dataset1", shape=(100,100,100), chunk_size=(32, 32, 32))
                    #print(dndataset)
                    dndataset = dosnatree.datasets.get(value, None)
                    print(dndataset)
                    #dndataset = dn.backends.base.BackendConnection.__getitem__(dosnatree, value)
                    if not value in list(h5file.keys()):
                        h5dataset = h5file.create_dataset(
                            key,
                            shape=dndataset.shape,
                            chunks=dndataset.chunk_size,
                            dtype=dndataset.dtype
                        )
                        for s in h5dataset.iter_chunks():
                            print(dndataset[s])
                            
                        #for b, s in zip(list(dndataset.data_chunks.keys()), h5dataset.iter_chunks()):
                        #    print(h5dataset[s])
                            
                    if dndataset is None:
                        raise Exception("No dataset found")
                    
            return h5file
        h5file = recurse(jsonfile, h)
        return h5file


cn = dn.Connection("dosna")
cn.connect()
first_tree = cn.create_tree("first_tree")
second_tree = first_tree.create_tree("second_tree")
dataset_one = first_tree.create_dataset("dataset1", shape=(100,100,100), chunk_size=(32, 32, 32))
x = dosnatohdf5(first_tree)
"""
x.dict_to_hdf5(first_tree.datasets)
"""
x.dosna_to_json()
json_dict = x.json_to_dict('file_to_write.json')
x.dict_to_hdf5(json_dict, first_tree, cn)