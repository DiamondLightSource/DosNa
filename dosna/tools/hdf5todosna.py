#!/usr/bin/env python
""" Tool to translate HDF5 files to DosNa"""

import logging

import uuid
import h5py
import json
import numpy as np

import dosna as dn
import dosna.tools.hdf5todict as hd
from dosna.tools.hdf5todict import LazyHdfDict #TODO: Create own methods?
import dosna.tools.dosnatohdf5 as Dosnatohdf5
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
        
        jsondict = _recurse(hdf5dict, {})
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

if __name__ == "__main__":

    def create_h5file(filename):
        with h5py.File(filename, "w") as f:
            A = f.create_group("A")
            B = A.create_group("B")
            C = A.create_group("C")
            D = B.create_group("D")

            A.attrs["a1"] = "One_value"
            A.attrs["a2"] = "Second_value"
            A.attrs["a3"] = "Third_value"
            A.attrs["a4"] = "Fourth_value"
            C.attrs["c1"] = "Value"

            dset1 = B.create_dataset("dset1", shape=(30,30))

            data = np.zeros((30, 30))
            for i in range(30):
                for j in range(30):
                    data[i][j] = j + 1
            dset1[...] = data

            dset2 = B.create_dataset("dset2", shape=(50, 50), chunks=(5,5))

            data = np.zeros((50, 50))
            for i in range(50):
                for j in range(50):
                    data[i][j] = j + 1
            dset2[...] = data

            dset3 = B.create_dataset("dset3", shape=(20,20), chunks=(2,2))

            data = np.zeros((20, 20))
            for i in range(20):
                for j in range(20):
                    data[i][j] = j + 1
            dset3[...] = data

    # CREATING FILE
    h5_filename = "testing_file.h5"
    h5_2_filename = "testing_2_file.h5"
    json_filename = "test_1.json"
    json_2_filename = "test_2.json"
    create_h5file(h5_filename)

    # DOSNA CONNECTION
    dn_connection = dn.Connection("dn-connection")

    # HDF5 TO DOSNA
    H5_TO_DN = Hdf5todosna(h5_filename)
    hdf5dict = H5_TO_DN.hdf5_to_dict()
    print("HDF5DICT ", hdf5dict, "\n")
    first_dosnadict = H5_TO_DN.hdf5dict_to_dosna(hdf5dict, dn_connection)
    print("FIRST DOSNADICT ", hdf5dict, "\n")
    jsondict = H5_TO_DN.hdf5dict_to_json(hdf5dict, json_filename)
    print("JSON DICT", jsondict, "\n")
    second_dosnadict = H5_TO_DN.json_to_dosna(json_filename, dn_connection)
    print("SECOND DOSNADICT", second_dosnadict, "\n")

    #print(dn_connection)
    #print("GROUP A", dn_connection.get_group("A"))
    print("======================================")

    # DOSNA TO HDF5
    DN_TO_H5 = Dosnatohdf5.Dosnatohdf5(dn_connection)
    third_dosnadict = DN_TO_H5.dosna_to_dict()
    print("THIRD DOSNADICT", third_dosnadict, "\n")
    DN_TO_H5.dosnadict_to_hdf5(third_dosnadict, h5_2_filename)
    second_jsondict = DN_TO_H5.dosnadict_to_jsondict(third_dosnadict, json_2_filename)
    print("SECOND JSONDICT", second_jsondict, "\n")
    DN_TO_H5.json_to_hdf5(json_filename, h5_2_filename)
    with h5py.File(h5_filename, "r") as f:
        print(f["A/B"].keys())

    with h5py.File(h5_2_filename, "r") as f:
        print(f["A/B"].keys())

    second_hdf5dict = DN_TO_H5.hdf5file_to_hdf5dict(h5_2_filename)
    print("HDF5DICT", second_hdf5dict)

