#!/usr/bin/env python

import logging as logging
import sys
import unittest

import numpy as np

import dosna as dn
from dosna.tests import configure_logger

log = logging.getLogger(__name__)


class DatasetTest(unittest.TestCase):
    """
    Test dataset actions
    """

    BACKEND = 'ram'
    ENGINE = 'cpu'
    CONNECTION_CONFIG = {}

    def setUp(self):
        self.handler = logging.StreamHandler(sys.stdout)
        log.addHandler(self.handler)
        log.info('DatasetTest: %s, %s, %s',
                 self.BACKEND, self.ENGINE, self.CONNECTION_CONFIG)

        dn.use(backend=self.BACKEND, engine=self.ENGINE)
        self.connection_handle = dn.Connection(**self.CONNECTION_CONFIG)
        self.connection_handle.connect()
        self.fake_dataset = 'NotADataset'
        self.data = np.random.rand(100, 100, 100)
        self.dataset = self.connection_handle.create_dataset(
            self.fake_dataset, data=self.data, chunks=(32, 32, 32))

    def tearDown(self):
        log.removeHandler(self.handler)
        if self.connection_handle.has_dataset(self.fake_dataset):
            self.connection_handle.del_dataset(self.fake_dataset)
        self.connection_handle.disconnect()

    def test_existing(self):
        self.assertTrue(self.connection_handle.has_dataset(self.fake_dataset))
        self.assertFalse(self.connection_handle.has_dataset(
            'NonExistantDataset'))

    def test_number_chunks(self):
        self.assertSequenceEqual(list(self.dataset.chunks), [4, 4, 4])

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

    DatasetTest.BACKEND = args.backend
    DatasetTest.ENGINE = args.engine
    DatasetTest.CONNECTION_CONFIG["name"] = args.connection
    DatasetTest.CONNECTION_CONFIG.update(
        dict(item.split('=') for item in args.connection_options))
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
