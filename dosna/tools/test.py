import h5py
import numpy as np
import dosna as dn
import hdf5todict.hdf5todict as hd

# NumPy arrays examples
a = np.arange(15)
b = np.array([7,8,9])
c = np.arange(10)

f = h5py.File("testlinks.h5", "w")

B = f.create_group("B")
Ba = f.create_group("Ba")
Bar = f.create_group("Bar")
Bazz = f.create_group("Bazz")
Battt = Bazz.create_group("Battt")
Bappp = Bazz.create_group("Bapppp")
Bafffff = Bazz.create_group("Bafffff")
Bak = Bafffff.create_group("Bak")
dset1 = Bak.create_dataset("dset1", shape=(2,2))
#Bak.attrs['shape'] = (2,3)
#Bak.attrs['comoquiersas'] = (34,3)
print(Bak.attrs)
"""
print(Bak.attrs.keys())
print(Bak.attrs['shape'])
print(dset1.attrs.keys())
"""
#print(Bazz.get("Bafffff/Bak")) #None



#print(f.keys())
#print(f.items())
#print(f.values())

cn = dn.Connection("dn-ram")
cn.connect()
print(cn.root_group)
print(cn.create_group("path"))
#A = cn.create_group("/")
#Bar = A.create_group("Bar")
#Bas = A.create_group("Bas")
#Baz = A.create_group("Baz")
#Car = Baz.create_group("Car")
#Dset = A.create_dataset("Dset1", shape=(2,2))

#print(A.visit())
#print(A.visititems())
#s = A.get_group("Baz/Car")
#t = A.has_group("Baz/Car")
#t = A.del_group("Baz")
#t = A.has_group("Baz")
#print(t.links)
#A.del_group("Baz/Car")
#A.get_group("Baz/Car")



import re
import dosna as dn
import numpy as np
import hdf5todict.hdf5todict as hd

cn = dn.Connection("dn-ram")
A = cn.create_node("location", "/")
B = A.create_node("B")
Ba = A.create_node("Ba")
Bar = A.create_node("Bar")
Bazz = A.create_node("Bazz")
Battt = Bazz.create_node("Battt")
Bappp = Bazz.create_node("Bapppp")
Bafffff = Bazz.create_node("Bafffff")
Dataset1 = Battt.create_node("Dataset1")
#print(A.iterate())
print(Bazz.iterate())
#print(A.get_node(""))
#print(Bazz.links.keys())
#print(A.iterate())
#A.visit()
#A.get_object_info()
print(A.get_node("Bazz/Battt/Dataset1"))
#print(Bazz.__contains__("Battfssft"))
#print(Bazz.keys())
#print(Bazz.values())
#print(Bazz.items())
#C = A.create_node("location", "C")
#D = B.create_node("location", "D")
#A.unlink("A", "C", "t")
#v = A.iterate()
#print(v)


cn = dn.Connection("dn-ram")
A = cn.create_node("location", "/")
B = A.create_node("B")
Ba = A.create_node("Ba")
Bar = A.create_node("Bar")
Bazz = A.create_node("Bazz")
Battt = Bazz.create_node("Battt")
Bappp = Bazz.create_node("Bapppp")
Bafffff = Bazz.create_node("Bafffff")
Dataset1 = Battt.create_node("Dataset1")
#print(A.iterate())
print(Bazz.iterate())
#print(A.get_node(""))
#print(Bazz.links.keys())
#print(A.iterate())
#A.visit()
#A.get_object_info()
print(A.get_node("Bazz/Battt/Dataset1"))
#print(Bazz.__contains__("Battfssft"))
#print(Bazz.keys())
#print(Bazz.values())
#print(Bazz.items())
#C = A.create_node("location", "C")
#D = B.create_node("location", "D")
#A.unlink("A", "C", "t")
#v = A.iterate()
#print(v)

#Ideal result
#<HDF5 group "/Bazz" (3 members)>

# Connection