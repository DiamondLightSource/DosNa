

import sys
import time
import logging
import unittest

import numpy as np


import dosna as dn


class DatasetTest(unittest.TestCase):

    def setUp(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        self.C = dn.Cluster().connect()
        self.test_pool_name = 'test_' + str(int(time.time()))
        self.C.create_pool(self.test_pool_name)
        self.test_pool = self.C[self.test_pool_name]

    def tearDown(self):
        self.test_pool.close()
        self.C.delete_pool(self.test_pool.name)
        self.C.disconnect()

    def test_create_dataset(self):
        self.test_pool.create_dataset('dummy', (10, 10, 10), np.float32)
        # Only one dataset has been created
        self.assertEqual(self.test_pool.dataset_count(), 1)
        # No chunk has been loaded yet, only 1 object should exist
        self.assertEqual(self.test_pool.object_count(), 1)
        # Check Existance of dataset
        self.assertTrue(self.test_pool.has_dataset('dummy'))
        self.assertFalse(self.test_pool.has_dataset('dummy2'))
        self.assertEqual(self.test_pool['dummy'].dtype, dn.dtype2str(np.float32))
        # Remove dataset
        self.test_pool.delete_dataset('dummy')
        self.assertEqual(self.test_pool.dataset_count(), 0)
        self.assertEqual(self.test_pool.object_count(), 0)

    def test_get_dataset(self):
        self.assertRaises(dn.DatasetException, dn.Dataset, self.test_pool, 'dummy2')
        try:
            self.test_pool.create_dataset('dummy2', data=np.random.randn(10, 10, 10))
            dn.Dataset(self.test_pool, 'dummy2')
            self.test_pool.delete_dataset('dummy2')
        except dn.DatasetException:
            self.fail('Exception shouldnt be thrown when dataset exists')

    def test_validate_dataset(self):
        data = np.arange(1000).reshape(10, 10, 10).astype(int)
        ds = self.test_pool.create_dataset('dummy', shape=data.shape, dtype=data.dtype,
                                           chunks=5)
        # Only one dataset has been created
        self.assertEqual(self.test_pool.dataset_count(), 1)
        # No chunk has been loaded yet, only 1 dataset should exist
        self.assertEqual(self.test_pool.object_count(), 1)
        # Get a random chunk
        chunk = ds.get_chunk(0, 0, 1)
        self.assertEqual(ds.dtype, dn.dtype2str(int))
        self.assertEqual(ds.dtype, chunk.dtype)
        # Because data has not been loaded yet, chunks will return the default value: -1
        np.testing.assert_array_equal(chunk[:], -1)
        # Chunks are dynamically created when readed/writed, thus 2 objects should exist now
        self.assertEqual(self.test_pool.dataset_count(), 1)
        self.assertEqual(self.test_pool.object_count(), 2)
        # Load data
        ds.load(data)
        self.assertEqual(self.test_pool.dataset_count(), 1)
        # Now 10/5 * 10/5 * 10/5 chunks should exist + 1 dataset object
        self.assertEqual(self.test_pool.object_count(), 2 * 2 * 2 + 1)
        # Validate the contents of the chunk
        # Note that the `chunk` object changes **in-place**. No need to retrieve it again
        np.testing.assert_array_equal(chunk[0, 0, :], [5, 6, 7, 8, 9])
        # Remove dataset
        ds.delete()
        self.assertEqual(self.test_pool.dataset_count(), 0)
        self.assertEqual(self.test_pool.object_count(), 0)

    def test_validate_large_chunks(self):
        data = np.arange(100*100*100).reshape(100, 100, 100).astype(int)
        ds = self.test_pool.create_dataset('dummy', data=data, chunks=50)
        # Now 10/5 * 10/5 * 10/5 chunks should exist + 1 dataset object
        self.assertEqual(self.test_pool.object_count(), 2 * 2 * 2 + 1)
        self.assertGreater(ds.chunk_bytes, 8192)
        # Delete dataset
        ds.delete()
        self.assertEqual(self.test_pool.object_count(), 0)
        # Create a new object with a single chunk -- default chunking
        ds = self.test_pool.create_dataset('dummy', data=data)
        # Now 1 chunk + 1 dataset object should exist
        self.assertEqual(self.test_pool.object_count(), 1 + 1)
        self.assertGreater(ds.chunk_bytes, 8192)
        self.assertEqual(ds.chunk_bytes, data.itemsize * data.size)
        # Check data fidelity
        np.testing.assert_array_equal(ds.get_chunk_data(0), data)
        # Delete
        ds.delete()

    def test_validate_slicing(self):
        data = np.random.randn(100, 100, 100)
        for chunks in np.random.randint(20, 100, size=10):
            ds = self.test_pool.create_dataset('dummy', data=data, chunks=chunks)
            self.check_slicing(ds, data)
            ds.delete()

    def check_slicing(self, dndata, npdata):
        # Invalid slices
        slice4d = (slice(None), slice(None), slice(None), slice(None))
        slicestep2 = slice(1, 10, 2)
        self.assertRaises(dn.DatasetException, dndata.__getitem__, 'm')
        self.assertRaises(dn.DatasetException, dndata.__getitem__, 7.5)
        self.assertRaises(dn.DatasetException, dndata.__getitem__, (7.5,))
        self.assertRaises(dn.DatasetException, dndata.__getitem__, slice4d)
        self.assertRaises(dn.DatasetException, dndata.__getitem__, slicestep2)
        # Valid slicing patterns
        np.testing.assert_array_equal(dndata[:], npdata[:])
        np.testing.assert_array_equal(dndata[50], npdata[50])
        np.testing.assert_array_equal(dndata[5:], npdata[5:])
        np.testing.assert_array_equal(dndata[5:12, 6:], npdata[5:12, 6:])
        np.testing.assert_array_equal(dndata[:, :5], npdata[:, :5])
        np.testing.assert_array_equal(dndata[...], npdata[...])
        np.testing.assert_array_equal(dndata[..., 6:10], npdata[..., 6:10])
        np.testing.assert_array_equal(dndata[..., 6:7, 8:9], npdata[..., 6:7, 8:9])
        np.testing.assert_array_equal(dndata[:5], npdata[:5])
        np.testing.assert_array_equal(dndata[:5, ...], npdata[:5, ...])
        np.testing.assert_array_equal(dndata[7:, ..., :5], npdata[7:, ..., :5])
        np.testing.assert_array_equal(dndata[50, 50, 50], npdata[50, 50, 50])
        np.testing.assert_array_equal(dndata[30:50, 30:50, 30:50], npdata[30:50, 30:50, 30:50])

    def test_validate_slice_writing(self):
        data = np.random.randn(100, 100, 100)
        for chunks in np.random.randint(20, 100, size=10):
            ds = self.test_pool.create_dataset('dummy', data=data, chunks=chunks)
            self.check_slice_writing(ds, data)
            ds.delete()

    def check_slice_writing(self, dndata, npdata, ntests=10):
        for i in range(ntests):
            slices = []
            sizes = []
            for shape in dndata.shape:
                size = np.random.randint(1, shape)
                start = np.random.randint(max(1, shape - size - 1))
                sizes.append(size)
                slices.append(slice(start, start+size))
            new_data = np.random.randn(*sizes)
            dndata[slices] = new_data
            npdata[slices] = new_data
            np.testing.assert_array_equal(dndata[slices], npdata[slices])


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("DatasetTest").setLevel(logging.DEBUG)
    unittest.main()
