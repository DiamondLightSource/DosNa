#!/usr/bin/env python
""" Tool to translate HDF5 files to DosNa"""

import logging

import numpy as np

import h5py
import json
import dosna as dn
import dosna.tools.hdf5todict.hdf5todict as hd
#import hdf5todict.hdf5todict as hd
from dosna.tools.hdf5todict.hdf5todict import LazyHdfDict #TODO: Create own methods?
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
        # TODO: check file is a h5file
        hdf5dict = hd.load(self._h5file, lazy=True)
        return hdf5dict
    
    def hdf5dict_to_dosna(self, hdf5dict, dosnaobject): # TODO: do I need the dosnadict at all?
        
        def _recurse(hdf5dict, dosnadict, group):
            for key, value in hdf5dict.items():
                if isinstance(value, LazyHdfDict):
                    dosnadict[key] = {}
                    group = group.create_group(key)
                    dosnadict[key] = _recurse(value, dosnadict[key], group)
                elif isinstance(value, h5py._hl.dataset.Dataset):
                    if value.nbytes > 10: # TODO: nbytes number
                        dataset = dosnaobject.create_dataset(
                            key,
                            shape=value.shape,
                            dtype=value.dtype,
                            chunk_size=value.chunks,
                        )
                        if value.chunks is not None:
                            for s in value.iter_chunks():
                                dataset[s] = value[s]
                        dosnadict[key] = dataset
                    else:
                        # TODO where does this go?
                        arr = np.zeros(value.shape)
                        value.read_direct(arr)
                        
            return dosnadict
        
        return _recurse(hdf5dict, {}, dosnaobject)


    def hdf5dict_to_jsondict(self, hdf5dict):
        
        def _recurse(hdf5dict, jsondict):
            for key, value in hdf5dict.items():
                jsondict[key] = {} # TODO: aqui?
                if isinstance(value, LazyHdfDict):
                    jsondict[key] = _recurse(value, jsondict[key])
                elif isinstance(value, h5py.Dataset):
                    # TODO: metadata jsondict[key]["name"] = value.name
                    jsondict[key] = value.name
            return jsondict
        
        return _recurse(hdf5dict, {})
    
    def jsondict_to_jsonfile(self, jsondict, jsonfile):
        with open(jsonfile, 'w') as f:
            f.write(json.dumps(jsondict))
        return jsonfile
    
    def jsonfile_to_jsondict(self, jsonfile):
        
        with open(jsonfile, 'r') as f:
            jsonstring = f.read()
        jsondict = json.loads(jsonstring)
        return jsondict

    def jsondict_to_dosna(self, jsondict, dosnaobject):

        def _recurse(jsondict, dosnadict):
            for key, value in jsondict.items():
                if isinstance(value, dict):
                    dosnadict[key] = {}
                    dosnadict[key][key] = dosnaobject.get_group(key)
                    dosnadict[key] = _recurse(value, dosnadict[key])
                elif isinstance(value, str):
                    dataset = dosnaobject.get_dataset(key)
                    dosnadict[key] = dataset
            return dosnadict
        
        return _recurse(jsondict, {})




"""
con = dn.Connection("dn-ssfadss")

# TESTING
x = Hdf5todosna('testfile.h5')
hdf5dict = x.hdf5_to_dict()
dosnadict = x.hdf5dict_to_dosna(hdf5dict, con)
#print(hdf5dict)
jsondict = x.hdf5dict_to_jsondict(hdf5dict)
jsonfile = x.jsondict_to_jsonfile(jsondict, 'testfile.json')
jsondict = x.jsonfile_to_jsondict(jsonfile)
dosnadict = x.jsondict_to_dosna(jsondict, con)
#print("===============")
#print(dosnadict)

#print(con.datasets)

import dosnatohdf5
name = "movefile5.h5"
y = dosnatohdf5.Dosnatohdf5(con)
jsondict = y.dosnadict_to_jsondict(dosnadict)
jsonfile = y.jsondict_to_jsonfile(jsondict, "test.json")
jsondict = y.jsonfile_to_jsondict(jsonfile)
h5file = y.jsondict_to_hdf5(jsondict, "testfile.h5")

#h = h5py.File(name, "r")
#print(h5file.keys())
#hfile = y.dosna_to_hdf5(dosnadict, name)

#print(hfile.keys())
f = h5py.File('testfile.h5', "r")
#print(f['bar'].keys())

import hdf5todosna

with h5py.File("testlinks.h5", "w") as f:
    B = f.create_group("B")
    dset1 = B.create_dataset("dset1", shape=(2,2))
    B.attrs['one_attribute'] = "Aqui hay un atributo"
    B.attrs['second_attribute'] = [2,3,4]
    B.attrs['third'] = "Aqui hay un atributo"
    dset1.attrs['shape'] = (34,3)
    dset1.attrs['dtype'] = "int"
    dset1.attrs['chunk_grid'] = np.asarray((2,2), dtype=int)
    dset1.attrs['chunk_size'] = np.asarray((4,4), dtype=int)
    #<Attributes of HDF5 object at 140573747160704>


"""
with h5py.File("try.h5", "w") as f:
    A = f.create_group("A")
    B = A.create_group("B")
    C = A.create_group("C")
    D = B.create_group("D")
    dset1 = B.create_dataset("dset1", shape=(2,2))
    
x = Hdf5todosna('try.h5')
con = dn.Connection("dn-csn")
hdf5dict = x.hdf5_to_dict()
#print(hdf5dict["B"]["dset1"].attrs['shape'])
dosnadict = x.hdf5dict_to_dosna(hdf5dict, con)
print(dosnadict)
#jsondict = x.hdf5dict_to_jsondict(hdf5dict)
#jsonfile = x.jsondict_to_jsonfile(jsondict, 'testlinks.json')
#jsondict = x.jsonfile_to_jsondict(jsonfile)
#dosnadict = x.jsondict_to_dosna(jsondict, con)
#print("===============")
#print(dosnadict)
