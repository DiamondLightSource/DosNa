

import sys
import numpy as np
import os.path as op

parent_folder = op.realpath(op.join(op.dirname(__file__), '..'))
sys.path.append(parent_folder)

import dosna as dn
from time_utils import Timer

conffile = op.join(parent_folder, 'ceph.conf')


data = np.random.randn(256, 256, 256)
data2 = np.random.randn(256, 256, 256)


for njobs in [1, 2, 4]:

    with Timer('Total time for tests'):
        with dn.Cluster(njobs=njobs) as C:
            print('\nTesting with {} thread(s)'.format(njobs))
            print('----------------------------------')
            pool = C.create_pool()

            pool.write_full('Warmup', 'Warming up!')
            pool.write_full('Warmup', 'Warming up!')
            pool.write_full('Warmup', 'Warming up!')

            with Timer('Dump data'):
                ds = pool.create_dataset('data', data=data, chunks=32)

            with Timer('Get all data'):
                cdata = ds[:]

            with Timer('Set all data'):
                ds[:] = data2

            with Timer('Get all data'):
                cdata2 = ds[:]

            with Timer('Apply addition'):
                ds.apply(lambda x: x + 1.0)

            with Timer('Get all data'):
                cdata3 = ds[:]

            with Timer('Map substraction'):
                ds2 = ds.map('data2', lambda x: x - 1.0)

            with Timer('Get all data'):
                cdata4 = ds2[:]

            np.testing.assert_allclose(data, cdata)
            np.testing.assert_allclose(data2, cdata2)
            np.testing.assert_allclose(data2 + 1, cdata3)
            np.testing.assert_allclose(data2, cdata4)
            print(cdata.dtype, cdata.shape)
