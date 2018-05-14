#!/usr/bin/env python

import logging as logging
import sys
import unittest

import numpy as np

import dosna as dn
from dosna.tests import configure_logger
from dosna.util.mpi import mpi_barrier
from dosna.util.mpi import mpi_comm
from dosna.util.mpi import mpi_is_root
from dosna.util.mpi import mpi_rank
from dosna.util.mpi import mpi_size

log = logging.getLogger(__name__)

BACKEND = 'ram'
ENGINE = 'cpu'
CONNECTION_CONFIG = {}

DATA_SIZE = (100, 100, 100)
DATA_CHUNK_SIZE = (32, 32, 32)

SEQUENTIAL_TEST_PARTS = 3
DATASET_NUMBER_RANGE = (-10000, 10000)


class MpiDatasetTest(unittest.TestCase):

    connection_handle = None

    @classmethod
    def setUpClass(cls):
        dn.use(backend=BACKEND, engine=ENGINE)
        cls.connection_handle = dn.Connection(**CONNECTION_CONFIG)
        cls.connection_handle.connect()

    @classmethod
    def tearDownClass(cls):
        cls.connection_handle.disconnect()
        cls.connection_handle = None

    def setUp(self):
        if ENGINE != "mpi" or mpi_size() < 2:
            self.skipTest("Test for engine mpi with several processes")

        if BACKEND == "ram":
            self.skipTest("Concurrent access in backend ram is not supported")

        log.info('DatasetTest: %s, %s, %s',
                 BACKEND, ENGINE, CONNECTION_CONFIG)

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
