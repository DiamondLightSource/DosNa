

import numpy as np
import dosna as dn

import os.path as op

conffile = op.realpath(op.join(op.dirname(__file__), '..', 'ceph.conf'))


# Safe access to a CEPH cluster
with dn.Cluster(conffile=conffile) as C:

    # Create pool and wrap with a safe context that opens and closes it as needed
    with C.create_pool('test_dosna_basic') as p:
        # Create a dataset with (32, 32, 32) chunking
        npdata = np.random.randint(100, size=(128, 128, 128))
        ds = p.create_dataset('data', data=npdata, chunks=32)
        print(ds.shape)

    # Accessing an existing poolm again with safe context
    with C['test_dosna_basic'] as p:
        npdata = np.random.randint(100, size=(16, 16, 16))
        ds = p.create_dataset('data2', data=npdata, chunks=2)
        print(ds.shape)

    # Open pool without context (remember to close it later)
    # Equivalent to dn.Pool('test_dosna_basic', cluster=C)
    p = C['test_dosna_basic']

    # Get an existing dataset
    ds = p['data2']

    # Stat a dataset -- shows number of bytes and creation date
    print("STAT", ds.stat())

    # Access dataset
    print(ds.shape, ds.dtype, ds.ndim)
    print(ds[:2, :2, :2])  # Supports numpy-like basic slicing (not advanced indexing)
    print(ds.get_chunk_data((0, 0, 0)))  # Equivalent to the above as chunks are 2x2x2
    # get_chunks supports linear indexing, check ds.total_chunks and ds.get_chunks
    print(ds.get_chunk_data(0))

    # Delete a dataset
    ds.delete()

    # Or delete it through the pool
    p.delete_dataset('data')

    # Don't forget to close the pool if accessed without context!!!
    p.close()

    # We can also delete a pool, it will delete all the files
    # Equivalent to C.delete_pool('test_dosna_basic')
    p.delete()



