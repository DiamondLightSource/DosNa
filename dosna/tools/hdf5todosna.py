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
        
        
        #Group B {datasets: {dset1: name: sfss}}
        #Group B {dset1: dset1, GroupC} # change datasets to the attribute # they have these attributes # whether alway sthe case
        
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
                    if value.nbytes < self.max_num_bytes:
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
                        pass # TODO 
                    else:
                        dosnadict[key] = {}
                        dosnaobject = dosnaobject.get_group(key)
                        if "datasets" in value:
                            for k, v in value["datasets"].items():
                                dataset  = dosnaobject.get_dataset(k)
                                #print(dosnaobject)
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
f = h5py.File("newfile.h5", "w")
f.close()
with h5py.File("newfile.h5", "w") as f:
    A = f.create_group("A")
    B = A.create_group("B")
    C = A.create_group("C")
    D = B.create_group("D")
    
    A.attrs["a1"] = "Otro_valor"
    A.attrs["a2"] = "Otro_valor"
    A.attrs["a3"] = "Otro_valor"
    A.attrs["a4"] = "Otro_valor"
    C.attrs["c1"] = "Valor"

    dset1 = B.create_dataset("dset1", shape=(2,2))
    dset2 = B.create_dataset("dset2", shape=(2,2), chunks=(1,1))
    dset3 = B.create_dataset("dset3", shape=(2,2), chunks=(1,1))
    
x = Hdf5todosna('newfile.h5')
con = dn.Connection("dn-csn")
hdf5dict = x.hdf5_to_dict()
dndict1 = x.hdf5dict_to_dosna(hdf5dict, con)
print("1", dndict1)
x.hdf5dict_to_json(hdf5dict, "mejor.json")
a = con.get_group("A")
b = a.get_group("B")
d = b.get_group("D")
print(b.links)
print(d.links)
#dndict2 = x.json_to_dosna("mejor.json", con)
#print("============")
#print(dndict2)
#a = con.get_group("A")
#print(a.get_groups())
#print(a.get_objects())
"""
#print(hdf5dict)
#print("====================")
#print("1", dndict1)
#print("=======================")
#print("2", dndict2)
#print(con.root_group.links) 


#print(a.name)
b = con.get_group("A/B")
a.get_group("B")
#print(b.name)
#print(b.get_group("dset1").name)
c = con.get_group("A/B/C")
#print(c.name)
d = con.get_group("A/B/D")

print(b.create_dataset("dset5", shape=(2,2)))
print(b.get_dataset("dset1"))
#print(d.name)
"""