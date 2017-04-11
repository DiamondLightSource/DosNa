

import sys
import numpy as np

if len(sys.argv) == 1:
    import h5py as dn
else:
    import dosna as dn
    dn.auto_init()

# Create File/Pool and wrap with a safe context that opens and closes it as needed
with dn.File('test_dosna_basic', 'w') as f:
    # Create a dataset with (32, 32, 32) chunking
    npdata = np.random.randint(100, size=(128, 128, 128))
    ds = f.create_dataset('data', data=npdata, chunks=(32, 32, 32))
    print(ds.shape)

# Accessing an existing File/Pool again with safe context
with dn.File('test_dosna_basic', 'a') as f:
    npdata = np.arange(16*16*16).reshape(16, 16, 16)
    ds = f.create_dataset('data2', data=npdata, chunks=(4, 4, 4))
    print(ds.shape)

# Open File/Pool without context (remember to close it later)
f = dn.File('test_dosna_basic', 'r')

# Get an existing dataset
ds = f['data2']

# Access dataset
print(ds.shape, ds.dtype, ds.ndim)
print(ds[:6, :6, :6])

# Don't forget to close the File if accessed without context!!!
f.close()

with dn.File('test_dosna_basic', 'a') as f:
    del f['data']
    del f['data2']