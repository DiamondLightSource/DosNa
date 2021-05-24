import h5py
import numpy as np
import dosna as dn
import hdf5todict.hdf5todict as hd


"""
HDF5 METHODS
"""
f = h5py.File("testlinks.h5", "w")

B = f.create_group("B")
Ba = f.create_group("Ba")
Bar = f.create_group("Bar")
Bazz = f.create_group("Baz")
Battt = Bazz.create_group("Batt")
Baww = Bazz.create_group("Baww")
dset1 = Baww.create_dataset("dset1", shape=(2,2))
Baww.attrs['shape'] = (2,3)
Baww.attrs['one_attribute'] = (34,3)
#print(Baww.attrs)
#print(Bazz.get("Bafffff/Bak")) #None

"""
DOSNA METHODS
"""
cn = dn.Connection("dn-ram")
cn.connect()
A = cn.create_group("A")
Bar = A.create_group("Bar")
Bas = A.create_group("Bas")
Baz = A.create_group("Baz")
Car = Baz.create_group("Car")
Kar = Car.create_group("Kar")
Dset = A.create_dataset("Dset1", shape=(2,2))


#print(A.keys())
#print(Bar.keys())
#print(Bas.keys())
#print(Baz.keys())
#print(Car.keys())
#print(Kar.keys())
#print(A.visit())
#print(A.visititems())
#s = A.get_group("Baz/Car")
#t = A.has_group("Baz/Car")
#t = A.del_group("Baz")
#t = A.has_group("Baz")
#print(t.links)
#A.del_group("Baz/Car")
#A.get_group("Baz/Car")