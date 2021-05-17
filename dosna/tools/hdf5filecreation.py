import h5py
import numpy as np

# NumPy arrays examples
a = np.arange(15)
b = np.array([7,8,9])
c = np.arange(10)

#Created the file
datatype = 'h5py.h5t.STD_I32BE'
f = h5py.File("testfile2.h5", "w")
dset1 = f.create_dataset("dset1", (10, 10), dtype=h5py.h5t.STD_I32BE, chunks=(2,2))

# Initialize data and write it to dset1
data = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        data[i][j] = j + 1	 
print("Writing data to dset1...")
dset1[...] = data

grp = f.create_group("bar")
dset2 = grp.create_dataset("dset2", (2, 10), dtype=h5py.h5t.STD_I32BE)

# Initialize data and write it to dset2.
data = np.zeros((2,10))
for i in range(2):
    for j in range(10):
        data[i][j] = j + 1	 
print("Writing data to dset2...")
dset2[...] = data

grp2 = f.create_group("bar2")
subgrp = grp.create_group("baz")
dset3 = subgrp.create_dataset("dset3", (2, 10), dtype=h5py.h5t.STD_I32BE)

# Initialize data and write it to dset2.
data = np.zeros((2,10))
for i in range(2):
    for j in range(10):
        data[i][j] = j + 1	 
print("Writing data to dset2...")
dset3[...] = data

subsubgrp = subgrp.create_group("bas")

#for s in dset.iter_chunks():
#    arr = dset[s]