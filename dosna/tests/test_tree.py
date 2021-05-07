#!/usr/bin/env python

import logging
import sys
import unittest

import numpy as np

import dosna as dn
from dosna.backends.base import DatasetNotFoundError
from dosna.tests import configure_logger


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

    connection_handle = None

    @classmethod
    def setUpClass(cls):
        dn.use(backend=BACKEND, engine=ENGINE)
        cls.connection_handle = dn.Connection(**CONNECTION_CONFIG)
        cls.connection_handle.connect()

    @classmethod
    def tearDownClass(cls):
        cls.connection_handle.disconnect()

    def setUp(self):
        if ENGINE == 'mpi':
            from dosna.util.mpi import mpi_size
            if mpi_size() > 1:
                self.skipTest("This should not test concurrent access")

        log.info('DatasetTest: %s, %s, %s',
                 BACKEND, ENGINE, CONNECTION_CONFIG)

        self.fake_dataset = 'NotADataset'
        self.data = np.random.random_integers(DATASET_NUMBER_RANGE[0],
                                              DATASET_NUMBER_RANGE[1],
                                              DATA_SIZE)
        self.tree = self.connection_handle.create_tree("hey")
        self.dataset = self.connection_handle.create_dataset(
            self.fake_dataset, data=self.data, chunk_size=DATA_CHUNK_SIZE)

    def tearDown(self):
        if self.connection_handle.has_dataset(self.fake_dataset):
            self.connection_handle.del_dataset(self.fake_dataset)
    
    def test_existing(self):
        self.assertTrue(self.connection_handle.has_dataset(self.fake_dataset))
        self.assertFalse(self.connection_handle.has_dataset(
            'NonExistantDataset'))

    def test_number_chunks(self):
        self.assertSequenceEqual(list(self.dataset.chunk_grid), [4, 4, 4])


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