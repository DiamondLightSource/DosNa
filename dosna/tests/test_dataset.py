#!/usr/bin/env python

import logging as logging
import sys
import unittest

import numpy as np

import dosna as dn
from dosna.backends.base import DatasetNotFoundError
from dosna.tests import configure_logger

try:
    from dosna.util.mpi import mpi_barrier
    from dosna.util.mpi import mpi_comm
    from dosna.util.mpi import mpi_is_root
    from dosna.util.mpi import mpi_rank
    from dosna.util.mpi import mpi_size
except ImportError:
    pass

log = logging.getLogger(__name__)

BACKEND = 'ram'
ENGINE = 'cpu'
CONNECTION_CONFIG = {}

DATA_SIZE = (100, 100, 100)
DATA_CHUNK_SIZE = (32, 32, 32)

SEQUENTIAL_TEST_PARTS = 3
DATASET_NUMBER_RANGE = (-10000, 10000)


class DatasetTest(unittest.TestCase):
    """
    Test dataset actions
    """

    def setUp(self):
        if ENGINE == 'mpi' and mpi_size() > 1:
            self.skipTest("This should not test concurrent access")

        log.info('DatasetTest: %s, %s, %s',
                 BACKEND, ENGINE, CONNECTION_CONFIG)

        dn.use(backend=BACKEND, engine=ENGINE)
        self.connection_handle = dn.Connection(**CONNECTION_CONFIG)
        self.connection_handle.connect()
        self.fake_dataset = 'NotADataset'
        self.data = np.random.random_integers(DATASET_NUMBER_RANGE[0],
                                              DATASET_NUMBER_RANGE[1],
                                              DATA_SIZE)
        self.dataset = self.connection_handle.create_dataset(
            self.fake_dataset, data=self.data, chunk_size=DATA_CHUNK_SIZE)

    def tearDown(self):
        if self.connection_handle.has_dataset(self.fake_dataset):
            self.connection_handle.del_dataset(self.fake_dataset)
        self.connection_handle.disconnect()

    def test_existing(self):
        self.assertTrue(self.connection_handle.has_dataset(self.fake_dataset))
        self.assertFalse(self.connection_handle.has_dataset(
            'NonExistantDataset'))

    def test_number_chunks(self):
        self.assertSequenceEqual(list(self.dataset.chunk_grid), [4, 4, 4])

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

        for slice_, expected in slices:
            slice_ = self.dataset._process_slices(slice_)
            iterator = self.dataset._chunk_slice_iterator(slice_,
                                                          self.dataset.ndim)
            self.assertEqual(len(list(iterator)), expected)

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

        for slice_ in slices:
            np.testing.assert_array_equal(self.dataset[slice_],
                                          self.data[slice_])

        np.testing.assert_array_equal(self.dataset[...], self.data)

    def test_dataset_clear(self):
        self.dataset.clear()
        np.testing.assert_array_equal(self.dataset[...],
                                      self.dataset.fillvalue)

    def test_map(self):
        dataset2 = self.dataset.map(lambda x: x + 1, self.fake_dataset + '2')
        np.testing.assert_array_equal(dataset2[...], self.dataset[...] + 1)
        self.connection_handle.del_dataset(dataset2.name)

    def test_apply(self):
        self.dataset.apply(lambda x: x + 1)
        np.testing.assert_array_equal(self.dataset[...], self.data + 1)

    def test_sequential_set(self):
        for part in range(SEQUENTIAL_TEST_PARTS):
            x_start = part * DATA_SIZE[0] // SEQUENTIAL_TEST_PARTS
            if part == SEQUENTIAL_TEST_PARTS - 1:
                x_stop = DATA_SIZE[0]
            else:
                x_stop = (part + 1) * DATA_SIZE[0] // SEQUENTIAL_TEST_PARTS
            self.dataset[x_start:x_stop] = \
                self.data[x_start:x_stop] * 3 + 5
        np.testing.assert_array_equal(self.dataset[...], self.data * 3 + 5)

    def test_del_non_existing_dataset(self):
        with self.assertRaises(DatasetNotFoundError):
            self.connection_handle.del_dataset('ThisDoesNotExist')

    def test_get_non_existing_dataset(self):
        with self.assertRaises(DatasetNotFoundError):
            self.connection_handle.get_dataset('ThisDoesNotExist')


class MpiDatasetTest(unittest.TestCase):

    def setUp(self):
        if ENGINE != "mpi" or mpi_size() < 2:
            self.skipTest("Test for engine mpi with several processes")

        if BACKEND == "ram":
            self.skipTest("Concurrent access in backend ram is not supported")

        log.info('DatasetTest: %s, %s, %s',
                 BACKEND, ENGINE, CONNECTION_CONFIG)

        dn.use(backend=BACKEND, engine=ENGINE)
        self.connection_handle = dn.Connection(**CONNECTION_CONFIG)
        self.connection_handle.connect()
        self.fake_dataset = 'NotADataset'
        data = None
        if mpi_is_root():
            data = np.random.random_integers(DATASET_NUMBER_RANGE[0],
                                             DATASET_NUMBER_RANGE[1],
                                             DATA_SIZE)
        self.data = mpi_comm().bcast(data, root=0)
        self.dataset = self.connection_handle.create_dataset(
            self.fake_dataset, data=self.data, chunk_size=DATA_CHUNK_SIZE)

    def tearDown(self):
        self.connection_handle.del_dataset(self.fake_dataset)
        self.connection_handle.disconnect()

    def test_load_function(self):
        np.testing.assert_array_equal(self.dataset[...], self.data)
        mpi_barrier()

    def test_non_overlapping_get(self):
        chunks_per_process = self.dataset.chunk_grid[0] // mpi_size()

        x_start = mpi_rank() * chunks_per_process * DATA_CHUNK_SIZE[0]
        if mpi_rank() == mpi_size() - 1:
            x_stop = self.dataset.shape[0]
        else:
            x_stop = (mpi_rank() + 1) * chunks_per_process * DATA_CHUNK_SIZE[0]
        if x_start < self.dataset.shape[0]:
            np.testing.assert_array_equal(self.dataset[x_start:x_stop, ...],
                                          self.data[x_start:x_stop, ...])
        mpi_barrier()

    def test_non_overlapping_set(self):
        chunks_per_process = self.dataset.chunk_grid[0] // mpi_size()

        x_start = mpi_rank() * chunks_per_process * DATA_CHUNK_SIZE[0]
        if mpi_rank() == mpi_size() - 1:
            x_stop = self.dataset.shape[0]
        else:
            x_stop = (mpi_rank() + 1) * chunks_per_process * DATA_CHUNK_SIZE[0]
        if x_start < self.dataset.shape[0]:
            self.dataset[x_start:x_stop, ...] = \
                self.data[x_start:x_stop, ...] * 3 + 5
        mpi_barrier()
        if mpi_is_root():
            expected = self.data * 3 + 5
            result = self.dataset[...]
            np.testing.assert_array_equal(result, expected)
        mpi_barrier()

    def test_dataset_clear(self):
        self.dataset.clear()
        if mpi_is_root():
            np.testing.assert_array_equal(self.dataset[...],
                                          self.dataset.fillvalue)
        mpi_barrier()

    def test_map(self):
        dataset2 = self.dataset.map(lambda x: x + 1, self.fake_dataset + '2')
        if mpi_is_root():
            np.testing.assert_array_equal(dataset2[...], self.dataset[...] + 1)
        mpi_barrier()
        dataset2.delete()

    def test_clone_and_load(self):
        clone_dataset = self.dataset.clone(
            "{}_clone".format(self.dataset.name))
        clone_dataset.load(self.dataset[...])
        if mpi_is_root():
            np.testing.assert_array_equal(clone_dataset[...], self.data)
        mpi_barrier()
        clone_dataset.delete()

    def test_apply(self):
        self.dataset.apply(lambda x: x + 1)
        if mpi_is_root():
            np.testing.assert_array_equal(self.dataset[...], self.data + 1)
        mpi_barrier()


def main():
    configure_logger(log)
    import argparse
    parser = argparse.ArgumentParser(description='TestDataset')
    parser.add_argument('--backend', dest='backend', default='ram',
                        help='Select backend (ram | hdf5 | ceph)')
    parser.add_argument('--engine', dest='engine', default='cpu',
                        help='Select engine (cpu | joblib | mpi)')
    parser.add_argument('--connection', dest='connection',
                        default='test-dosna',
                        help='Connection name')
    parser.add_argument('--connection-options', dest='connection_options',
                        nargs='+', default=[],
                        help='Connection options using the format: '
                             'key1=val1 [key2=val2...]')

    args, unknown_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown_args

    global BACKEND, ENGINE, CONNECTION_CONFIG
    BACKEND = args.backend
    ENGINE = args.engine
    CONNECTION_CONFIG["name"] = args.connection
    CONNECTION_CONFIG.update(
        dict(item.split('=') for item in args.connection_options))
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
