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

print(Bazz.get("Bafffff/Bak")) #None



#print(f.keys())
#print(f.items())
#print(f.values())

cn = dn.Connection("dn-ram")
cn.connect()
A = cn.create_group("/")
Bazz = A.create_group("Bazz")
Batt = Bazz.create_group("Batt")
Dset = A.create_dataset("Dset1", shape=(2,2))

print(A.keys())
print(A.values())
print(A.items())

