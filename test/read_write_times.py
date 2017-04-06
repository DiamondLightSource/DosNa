

NTEST = 3
CHUNKS = (2, 4, 8, 16, 32, 64, 128, 256, 512)


import sys
import os.path as op
import time

parent_folder = op.realpath(op.join(op.dirname(__file__), '..'))
sys.path.append(parent_folder)

import numpy as np
import dosna as dn

with dn.Cluster(conffile=op.join(parent_folder, 'ceph.conf')) as C:
    test_pool_name = "test_" + str(time.time())
    pool = C.create_pool(test_pool_name)

    npdata = np.random.randn(1000, 1000, 1000)

    timings = np.zeros((len(CHUNKS), NTEST, 3), np.float32)

    for i, chunk_size in enumerate(CHUNKS):
        print "CHUNKS", chunk_size
        for j in range(NTEST):
            print('Dumping data onto memory')
            t0 = time.time()
            dsdata = dn.Dataset(pool, 'data', data=npdata, chunks=chunk_size)
            t1 = time.time()
            timings[i, j, 0] = t1 - t0

            print('Reading all the data')
            t0 = time.time()
            dsdata[:]
            t1 = time.time()
            timings[i, j, 1] = t1 - t0

            #print('Replacing all the data')
            #t0 = time.time()
            #dsdata[:] = npdata2
            #t1 = time.time()
            #timings[i, j, 2] = t1 - t0

            print('Deleting dataset')
            t0 = time.time()
            dsdata.delete()
            t1 = time.time()
            timings[i, j, 2] = t1 - t0

    np.save(op.join(parent_folder, 'timings.npy'), timings)

    pool.close()
    pool.delete()

print timings