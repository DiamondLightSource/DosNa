#!/usr/bin/env python
""" Tool to translate HDF5 files to DosNa"""

import logging

import numpy as np

import h5py
import json
import dosna as dn
import dosna.tools.hdf5todict.hdf5todict as hd
from dosna.tools.hdf5todict.hdf5todict import LazyHdfDict #TODO: Create own methods?
from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset, DatasetNotFoundError,
                                 BackendGroup, GroupNotFoundError)
from h5py._hl.dataset import Dataset

MAX_NUM_BYTES = 20

class Hdf5todosna():
    """
    Includes methods to transform HDF5 to DosNa Objects
    """

    def __init__(self, h5file=None, *args, **kwargs):
        self._h5file = h5file

    def hdf5_to_dict(self):
        hdf5dict = hd.load(self._h5file, lazy=True)
        return hdf5dict
    
    def hdf5dict_to_dosna(self, hdf5dict, dosnaobject): # TODO: do I need the dosnadict at all?
        
        def _recurse(hdf5dict, dosnadict, group):
            for key, value in hdf5dict.items():
                if isinstance(value, LazyHdfDict):
                    dosnadict[key] = {}
                    group = group.create_group(key)
                    for k, v in value.items():
                        if isinstance(v, Dataset):
                            ds = group.create_dataset(
                            k,
                            shape=v.shape,
                            dtype=v.dtype,
                            chunk_size=v.chunks,
                        )
                            print(group, group.name)
                            print(ds)
                            dosnadict[key][k] = ds   
                            print(dosnadict[key][k])
                    dosnadict[key] = _recurse(value, dosnadict[key], group)
                elif isinstance(value, Dataset):
                    if value.nbytes > 5: # TODO: nbytes number
                        # TODO change duplicado
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
                        # TODO where does this go?
                        arr = np.zeros(value.shape)
                        value.read_direct(arr)
                        
            return dosnadict
        dosnadict = _recurse(hdf5dict, {}, dosnaobject)
        return dosnadict 


    def hdf5dict_to_json(self, hdf5dict, jsonfile): #TODO specify which type of json you are returning
        
        def _recurse(hdf5dict, jsondict):
            for key, value in hdf5dict.items():
                if isinstance(value, LazyHdfDict):
                    jsondict[key] = {}
                    jsondict[key] = _recurse(value, jsondict[key])
                elif isinstance(value, Dataset):
                    if not "datasets" in jsondict:
                        jsondict["datasets"] = {}
                    jsondict["datasets"][key] = {}
                    jsondict["datasets"][key]["name"] = key # TODO path = key.split("/")
                    jsondict["datasets"][key]["shape"] = value.shape
                    jsondict["datasets"][key]["ndim"] = value.ndim
                    jsondict["datasets"][key]["dtype"] = value.dtype.str 
                    jsondict["datasets"][key]["nbytes"] = value.nbytes.item()
                    jsondict["datasets"][key]["is_dataset"] = True
                    jsondict["datasets"][key]["absolute_path"] = value.name
                    if value.nbytes < MAX_NUM_BYTES:
                        data_np = np.zeros(value.shape)
                        value.read_direct(data_np)
                        jsondict["datasets"][key]["value"] = data_np.tolist()
                        
            return jsondict
        
        jsondict = _recurse(hdf5dict, {})
        
        with open(jsonfile, 'w') as f:
            f.write(json.dumps(jsondict))
        
    
    def json_to_dosna(self, jsonfile, dosnaobject):
        with open(jsonfile, 'r') as f:
            jsonstring = f.read()
        jsondict = json.loads(jsonstring)
        def _recurse(jsondict, dosnadict, dosnaobject):
            for key, value in jsondict.items():
                if isinstance(value, dict):

                    if jsondict[key].get("is_dataset") or key=="datasets":
                        pass
                    else:
                        dosnadict[key] = {}
                        dosnaobject = dosnaobject.get_group(key)
                        if "datasets" in value:
                            for k, v in value["datasets"].items():
                                dataset  = dosnaobject.get_group(k)
                                dosnadict[key][k] = dataset
                        dosnadict[key] = _recurse(value, dosnadict[key], dosnaobject)
            return dosnadict
        
        return _recurse(jsondict, {}, dosnaobject)




"""
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
    dset2 = B.create_dataset("dset2", shape=(2,2), chunks=(1,1))
    dset3 = B.create_dataset("dset3", shape=(2,2), chunks=(1,1))
    
x = Hdf5todosna('try.h5')
con = dn.Connection("dn-csn")
ss = con.create_group("ss")
hdf5dict = x.hdf5_to_dict()
dndict1 = x.hdf5dict_to_dosna(hdf5dict, con)
#print("1", dndict1)
x.hdf5dict_to_json(hdf5dict, "mejor.json")
dndict2 = x.json_to_dosna("mejor.json", con)
print("1", dndict1)
print("2", dndict2)
#print(con.root_group.links) 
a = con.get_group("A")

#print(a.name)
b = con.get_group("A/B")
a.get_group("B")
#print(b.name)
#print(b.get_group("dset1").name)
c = con.get_group("A/B/C")
#print(c.name)
d = con.get_group("A/B/D")
#print(d.name)
