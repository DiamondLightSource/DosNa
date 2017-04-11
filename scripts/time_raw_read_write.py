

import os
import sys
import numpy as np
import os.path as op

parent_folder = op.realpath(op.join(op.dirname(__file__), '..'))
sys.path.append(parent_folder)

import rados
import dosna as dn
from script_utils import Timer

conffile = op.join(parent_folder, 'ceph.conf')

###################

data = np.random.randn(128, 128, 128)


with Timer('Numpy to bytes'):
    bytes = data.tobytes()

###################

print('\nTesting raw RADOS')
print('----------------------')

cluster = rados.Rados(conffile=conffile)
cluster.connect()

cluster.create_pool('test_dosna_dummy')

ioctx = cluster.open_ioctx('test_dosna_dummy')

ioctx.write_full('hello', 'warm up')

with Timer('Write data Rados'):
    ioctx.write_full('dummy', bytes)

ioctx.close()

cluster.delete_pool('test_dosna_dummy')
cluster.shutdown()


###################

print('\nTesting DosNa interface')
print('----------------------')

with dn.Cluster(conffile=conffile) as C:
    pool = C.create_pool(test=True)

    pool.write_full('hello', 'warm up')

    with Timer('Write low-level interface DosNa'):
        pool.write_full('dummy', bytes)

    with Timer('Write DataChunk DosNa'):
        dn.DataChunk.create(pool, 'dummy', data=data)

    pool.delete()

###################

print('\nTesting HDF5')
print('----------------------')

import h5py as h5

with h5.File('/tmp/dummy_dosna_test.h5', 'w') as f:
    with Timer('Write raw data'):
        f.create_dataset('dummy', data=data)
        f.flush()

os.remove('/tmp/dummy_dosna_test.h5')