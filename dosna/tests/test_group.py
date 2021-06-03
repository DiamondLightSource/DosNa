#!/usr/bin/env python

import logging
import sys
import unittest

import numpy as np

import dosna as dn
from dosna.backends.base import DatasetNotFoundError, GroupNotFoundError
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
        log.info('GroupTest: %s, %s, %s',
                 BACKEND, ENGINE, CONNECTION_CONFIG)

        self.fake_group = 'NotAGroup'
        self.group = self.connection_handle.create_group(self.fake_group)

    def tearDown(self):
        if self.connection_handle.has_group(self.fake_group):
            self.connection_handle.del_group(self.fake_group)

    def test_existing_group(self):
        self.assertTrue(self.connection_handle.has_group(self.fake_group))

    def test_create_group_same_name(self):
        group_name = "NewGroup"
        self.connection_handle.create_group(group_name)
        with self.assertRaises(Exception):
            self.connection_handle.create_group(group_name)
        self.connection_handle.del_group(group_name)

    def test_create_group_not_alphanumeric(self):
        group_name = "/NotAlphanumeric?"
        with self.assertRaises(Exception):
            self.connection_handle.create_group(group_name)

    def test_del_group_create_same_group(self):
        self.connection_handle.del_group(self.fake_group)
        self.assertIsNotNone(self.connection_handle.create_group(self.fake_group))

    def test_del_group_links(self):
        self.group.create_group("Subgroup")
        self.assertIn("Subgroup", self.group.links)
        self.group.del_group("Subgroup")
        self.assertNotIn("Subgroup", self.group.links)

    def test_get_group_by_path(self):
        group_A = self.connection_handle.create_group("A")
        group_B = group_A.create_group("B")
        group_C = group_B.create_group("C")
        self.assertIsNotNone(group_A.get_group("B"))
        self.assertIsNotNone(group_A.get_group("B/C"))

    def test_update_metadata(self):
        self.assertEqual({}, self.group.get_metadata())
        self.group.add_metadata({'Key1': 'Value1'})
        self.assertEqual({'Key1': 'Value1'}, self.group.get_metadata())
        self.group.add_metadata({'Key2': 'Value2'})
        self.assertEqual({'Key1': 'Value1', 'Key2': 'Value2'}, self.group.get_metadata())

    def test_not_existing_group(self):
        with self.assertRaises(GroupNotFoundError):
            self.connection_handle.has_group('NonExistingGroup')

    def test_del_non_existing_group(self):
        with self.assertRaises(GroupNotFoundError):
            self.connection_handle.del_group('NonExistingGroup')

    def test_get_non_existing_group(self):
        with self.assertRaises(GroupNotFoundError):
            self.connection_handle.get_group('NonExistingGroup')

    def test_del_non_existing_dataset(self):
        with self.assertRaises(DatasetNotFoundError):
            self.group.del_dataset('ThisDoesNotExist')

    def test_get_non_existing_dataset(self):
        with self.assertRaises(DatasetNotFoundError):
            self.group.get_dataset('ThisDoesNotExist')


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