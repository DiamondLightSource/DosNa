

import sys
import unittest
import logging as logging

import numpy as np
import dosna as dn

logging.basicConfig(level=logging.DEBUG, format="LOG: %(message)s")
log = logging.getLogger()
log.level = logging.INFO


class PoolTest(unittest.TestCase):

    BACKEND = 'ram'
    ENGINE = 'cpu'
    CONFIG = None
    POOL = 'test_dosna'

    def setUp(self):
        self.handler = logging.StreamHandler(sys.stdout)
        log.addHandler(self.handler)
        log.info('DatasetTest: {}, {}, {}'
                 .format(self.BACKEND, self.ENGINE, self.CONFIG))

        dn.use(backend=self.BACKEND, engine=self.ENGINE)
        self.C = dn.Cluster(self.CONFIG)
        if self.BACKEND != 'ceph' and not self.C.has_pool(self.POOL):
            # Create pool for 'ram' and 'hdf5' backends
            self.C.create_pool(self.POOL)
        self.pool = self.C[self.POOL]
        self.fakeds = 'NotADataset'
        self.data = np.random.rand(100, 100, 100)
        self.ds = self.pool.create_dataset(
            self.fakeds, data=self.data, chunks=(32, 32, 32))

    def tearDown(self):
        log.removeHandler(self.handler)
        if self.BACKEND != 'ceph':  # Avoid creating pools in ceph
            self.C.del_pool(self.POOL)
        if self.pool.has_dataset(self.fakeds):
            self.pool.del_dataset(self.fakeds)
        self.C.disconnect()

    def test_existing(self):
        self.assertTrue(self.pool.has_dataset(self.fakeds))

    def test_number_chunks(self):
        self.assertSequenceEqual(list(self.ds.chunks), [4, 4, 4])

    def test_number_chunks_slicing(self):
        slices = [
            #  50, :32, :32 = 1 * 1 * 1
            [(50, slice(0, 32), slice(0, 32)), 1 * 1 * 1],
            #  50, :32, :33 = 1 * 1 * 2
            [(50, slice(0, 32), slice(0, 33)), 1 * 1 * 2],
            # :50, :32, :33 = 2 * 1 * 2
            [(slice(0, 50), slice(0, 32), slice(0, 33)), 2 * 1 * 2],
            # :, :66, 50:96 = 4 * 3 * 2
            [(slice(None), slice(0, 66), slice(50, 96)), 4 * 3 * 2],
            # ... = 4 * 4 * 4
            [slice(None), 4 * 4 * 4]
        ]

        for sl, expected in slices:
            sl = self.ds._process_slices(sl)
            it = self.ds._chunk_slice_iterator(sl, self.ds.ndim)
            self.assertEqual(len(list(it)), expected)

    def test_slice_content(self):
        slices = [
            #  50, :32, :32
            (50, slice(0, 32), slice(0, 32)),
            #  50, :32, :33
            (50, slice(0, 32), slice(0, 33)),
            # :50, :32, :33
            (slice(0, 50), slice(0, 32), slice(0, 33)),
            # :, :66, 50:96
            (slice(None), slice(0, 66), slice(50, 96)),
            # ... = 4 * 4 * 4
            slice(None)
        ]

        for sl in slices:
            np.testing.assert_array_equal(self.ds[...], self.data)

    def test_dataset_clear(self):
        self.ds.clear()
        np.testing.assert_array_equal(self.ds[...], self.ds.fillvalue)

    def test_map(self):
        ds2 = self.ds.map(lambda x: x + 1, self.fakeds + '2')
        np.testing.assert_array_equal(ds2[...], self.ds[...] + 1)
        self.pool.del_dataset(ds2.name)

    def test_apply(self):
        self.ds.apply(lambda x: x + 1)
        np.testing.assert_array_equal(self.ds[...], self.data + 1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        PoolTest.BACKEND = sys.argv.pop(1)
    if len(sys.argv) > 1:
        PoolTest.ENGINE = sys.argv.pop(1)
    if len(sys.argv) > 1:
        PoolTest.CONFIG = sys.argv.pop(1)
    if len(sys.argv) > 1:
        PoolTest.POOL = sys.argv.pop(1)

    unittest.main(verbosity=2)