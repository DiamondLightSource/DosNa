#!/usr/bin/env python

import sys
import unittest
import logging as logging

import numpy as np
import dosna as dn

logging.basicConfig(level=logging.DEBUG, format="LOG: %(message)s")
log = logging.getLogger(__name__)
log.level = logging.INFO


class DatasetTest(unittest.TestCase):

    BACKEND = 'ram'
    ENGINE = 'cpu'
    POOL = 'test_dosna'
    CLUSTER_CONFIG = {}

    def setUp(self):
        self.handler = logging.StreamHandler(sys.stdout)
        log.addHandler(self.handler)
        log.info('DatasetTest: {}, {}, {}'
                 .format(self.BACKEND, self.ENGINE, self.CLUSTER_CONFIG))

        dn.use(backend=self.BACKEND, engine=self.ENGINE)
        self.C = dn.Connection(**self.CLUSTER_CONFIG)
        self.C.connect()
        self.fakeds = 'NotADataset'
        self.data = np.random.rand(100, 100, 100)
        self.ds = self.C.create_dataset(
            self.fakeds, data=self.data, chunks=(32, 32, 32))

    def tearDown(self):
        log.removeHandler(self.handler)
        if self.C.has_dataset(self.fakeds):
            self.C.del_dataset(self.fakeds)
        self.C.disconnect()

    def test_existing(self):
        self.assertTrue(self.C.has_dataset(self.fakeds))

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
        self.C.del_dataset(ds2.name)

    def test_apply(self):
        self.ds.apply(lambda x: x + 1)
        np.testing.assert_array_equal(self.ds[...], self.data + 1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='TestDataset')
    parser.add_argument('--backend', dest='backend', default='ram',
                        help='Select backend (ram | hdf5 | ceph)')
    parser.add_argument('--engine', dest='engine', default='cpu',
                        help='Select engine (cpu | joblib | mpi)')
    parser.add_argument('--cluster', dest='cluster', default='test-cluster',
                        help='Cluster name')
    parser.add_argument('--cluster-options', dest='cluster_options', nargs='+',
                        default=[], help='Cluster options using the format: '
                                         'key1=val1 [key2=val2...]')
    parser.add_argument('--pool', dest='pool', default='test_dosna',
                        help='Existing pool to use during tests '
                        '(default: test_dosna).')

    args, unknownargs = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknownargs

    DatasetTest.BACKEND = args.backend
    DatasetTest.ENGINE = args.engine
    DatasetTest.POOL = args.pool
    DatasetTest.CLUSTER_CONFIG["name"] = args.cluster
    DatasetTest.CLUSTER_CONFIG.update(
        dict(item.split('=') for item in args.cluster_options))

    unittest.main(verbosity=2)
