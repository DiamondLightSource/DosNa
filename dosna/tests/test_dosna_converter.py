import logging
import sys
import unittest
import os
import h5py
import numpy as np
from numpy.testing import assert_array_equal

import dosna as dn
from dosna.tests import configure_logger

from dosna.tools.dosna_converter import Dosnatohdf5

log = logging.getLogger(__name__)

BACKEND = 'ram'
ENGINE = 'cpu'
CONNECTION_CONFIG = {}

DATA_SIZE = (100, 100, 100)
DATA_CHUNK_SIZE = (32, 32, 32)

SEQUENTIAL_TEST_PARTS = 3
DATASET_NUMBER_RANGE = (-10000, 10000)

H5FILE_NAME = 'test_h5_file.h5'
DATASET_NAME = 'fakedataset'
JSON_FILE_NAME = 'test_json_file.json'


class Hdf5todosnaTest(unittest.TestCase):
    """
    Test HDF5 to DosNa methods
    """
    @classmethod
    def setUpClass(cls):
        dn.use(backend=BACKEND, engine=ENGINE)
        cls.dn_connection = dn.Connection(**CONNECTION_CONFIG)
        cls.dn_connection.connect()
        cls.dn_converter = Dosnatohdf5(cls.dn_connection)

    @classmethod
    def tearDownClass(cls):
        cls.dn_connection.disconnect()

    def setUp(self):

        group_A = self.dn_connection.create_group("A", {'A1': 'V1', 'A2': 'V2'})
        group_B = group_A.create_group("B")
        group_C = group_A.create_group("C", {'C1': 'V1'})
        group_D = group_B.create_group("D")

        data1 = np.random.randint(DATASET_NUMBER_RANGE[0], DATASET_NUMBER_RANGE[1]+1, DATA_SIZE)
        dset1 = group_B.create_dataset('dset1', data=data1)
        data2 = np.random.randint(DATASET_NUMBER_RANGE[0], DATASET_NUMBER_RANGE[1] + 1, DATA_SIZE)
        dset2 = group_B.create_dataset('dset2', data=data2, chunk_size=DATA_CHUNK_SIZE)



    def tearDown(self):
        if self.dn_connection.has_group('A'):
            group_A = self.dn_connection.get_group('A')
            group_B = group_A.get_group('B')
            group_C = group_A.get_group('C')
            group_D = group_B.get_group('D')

            group_B.del_group('D')
            group_A.del_group('C')
            group_A.del_group('B')
            self.dn_connection.del_group('A')
        self.dn_connection.disconnect()
        if os.path.isfile(H5FILE_NAME):
            os.remove(H5FILE_NAME)
        if os.path.isfile(JSON_FILE_NAME):
            os.remove(JSON_FILE_NAME)

    def check_keys(self, expected_key, dictionary):
        idx = 0
        for key in dictionary:
            if key == "attrs":
                continue
            self.assertEqual(list(expected_key)[idx], key)
            idx += 1


    def compare_datasets_dosna(self, dset1, dset2):
        self.assertEqual(dset1.name, dset2.name)
        self.assertEqual(dset1.get_absolute_path(), dset2.get_absolute_path())
        self.assertEqual(dset1.shape, dset2.shape)
        self.assertEqual(dset1.dtype, dset2.dtype)
        self.assertEqual(dset1.ndim, dset2.ndim)
        self.assertEqual(dset1.fillvalue, dset2.fillvalue)
        self.assertIsNone(np.testing.assert_array_equal(dset1.chunk_size, dset2.chunk_size))
        self.assertIsNone(np.testing.assert_array_equal(dset1.chunk_grid, dset2.chunk_grid))
        self.assertIsNone(np.testing.assert_array_equal(dset1.total_chunks, dset2.total_chunks))
        for hdf, dn in zip(dset1, dset2):
            self.assertIsNone(assert_array_equal(hdf, dn))
        for i in range(0, dset1.total_chunks):
            idx = dset1._idx_from_flat(i)
            self.assertIsNone(assert_array_equal(dset1[idx], dset2[idx]))

    def compare_datasets_hdf(self, dn_dset, hdf_dset):
        self.assertEqual(dn_dset.get_absolute_path(), hdf_dset.name)
        self.assertEqual(hdf_dset.shape, dn_dset.shape)
        self.assertEqual(hdf_dset.dtype, dn_dset.dtype)
        self.assertEqual(hdf_dset.ndim, dn_dset.ndim)
        self.assertEqual(hdf_dset.fillvalue, dn_dset.fillvalue)
        for hdf, dn in zip(hdf_dset, dn_dset):
            self.assertIsNone(assert_array_equal(hdf, dn))
        if hdf_dset.chunks is not None:
            self.assertEqual(hdf_dset.chunks, dn_dset.chunk_size)
            for i in range(0, dn_dset.total_chunks):
                idx = dn_dset._idx_from_flat(i)
                self.assertIsNone(assert_array_equal(dn_dset[idx], hdf_dset[idx]))

    def compare_datasets_json(self, dn_dset, json_dset):
        self.assertEqual(dn_dset.get_absolute_path(), json_dset["absolute_path"])
        self.assertEqual(dn_dset.name, json_dset["name"])
        self.assertEqual(dn_dset.shape, json_dset["shape"])
        self.assertEqual(dn_dset.dtype, json_dset["dtype"])
        self.assertEqual(dn_dset.ndim, json_dset["ndim"])
        self.assertEqual(dn_dset.fillvalue, json_dset["fillvalue"])
        self.assertEqual(dn_dset.chunk_size, json_dset["chunk_size"])
        self.assertIsNone(np.testing.assert_array_equal(dn_dset.chunk_grid, json_dset["chunk_grid"]))
        for d1, d2 in zip(dn_dset, json_dset['dataset_value']):
            self.assertIsNone(assert_array_equal(d1, d2))


    def test_dosna2dict(self):
        dosna_dict = self.dn_converter.dosna2dict()
        dn_cluster = self.dn_connection

        self.assertDictEqual(dn_cluster['A'].attrs, dosna_dict['A']['attrs'])
        self.assertEqual(dn_cluster['A']['B'].attrs, dosna_dict['A']['B']['attrs'])
        self.assertEqual(dn_cluster['A']['B']['D'].attrs, dosna_dict['A']['B']['D']['attrs'])
        self.assertEqual(dn_cluster['A']['C'].attrs, dosna_dict['A']['C']['attrs'])

        self.check_keys(dn_cluster['A'].keys(), dosna_dict['A'])
        self.check_keys(dn_cluster['A']['B'].keys(), dosna_dict['A']['B'])
        self.check_keys(dn_cluster['A']['B']['D'].keys(), dosna_dict['A']['B']['D'])
        self.check_keys(dn_cluster['A']['C'].keys(), dosna_dict['A']['C'])

        dn_cluster_dset1 = dn_cluster['A']['B']['dset1']
        dosna_dict_dset1 = dosna_dict['A']['B']['dset1']
        self.compare_datasets_dosna(dn_cluster_dset1, dosna_dict_dset1)

        dn_cluster_dset2 = dn_cluster['A']['B']['dset2']
        dosna_dict_dset2 = dosna_dict['A']['B']['dset2']
        self.compare_datasets_dosna(dn_cluster_dset2, dosna_dict_dset2)

    def test_dosna2hdf(self):
        self.dn_converter.dosna2hdf(H5FILE_NAME)
        hdf_file = h5py.File(H5FILE_NAME)
        dn_cluster = self.dn_connection

        hdf_file_attrs = dict(attr for attr in hdf_file['A'].attrs.items())
        self.assertDictEqual(hdf_file_attrs, dn_cluster['A'].attrs)
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['B'].attrs.items())
        self.assertEqual(hdf_file_attrs, dn_cluster['A']['B'].attrs)
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['B']['D'].attrs.items())
        self.assertEqual(hdf_file_attrs, dn_cluster['A']['B']['D'].attrs)
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['C'].attrs.items())
        self.assertEqual(hdf_file_attrs, dn_cluster['A']['C'].attrs)

        self.assertEqual(set(hdf_file.keys()), set(dn_cluster.root_group.keys()))
        self.assertEqual(set(hdf_file['A'].keys()), set(dn_cluster['A'].keys()))
        self.assertEqual(set(hdf_file['A']['B'].keys()), set(dn_cluster['A']['B'].keys()))
        self.assertEqual(set(hdf_file['A']['B']['D'].keys()), set(dn_cluster['A']['B']['D'].keys()))
        self.assertEqual(set(hdf_file['A']['C'].keys()), set(dn_cluster['A']['C'].keys()))

        hdf_file_dset1 = hdf_file['A']['B']['dset1']
        dn_cluster_dset1 = dn_cluster['A']['B']['dset1']
        self.compare_datasets_hdf(dn_cluster_dset1, hdf_file_dset1)

        hdf_file_dset2 = hdf_file['A']['B']['dset2']
        dn_cluster_dset2 = dn_cluster['A']['B']['dset2']
        self.compare_datasets_hdf(dn_cluster_dset2, hdf_file_dset2)
        hdf_file.close()

    def test_dosna2json(self):
        json_dict = self.dn_converter.dosna2json(JSON_FILE_NAME)
        dn_cluster = self.dn_connection

        self.assertDictEqual(dn_cluster['A'].attrs, json_dict['A']['attrs'])
        self.assertEqual(dn_cluster['A']['B'].attrs, json_dict['A']['B']['attrs'])
        self.assertEqual(dn_cluster['A']['B']['D'].attrs, json_dict['A']['B']['D']['attrs'])
        self.assertEqual(dn_cluster['A']['C'].attrs, json_dict['A']['C']['attrs'])

        self.check_keys(dn_cluster['A'].keys(), json_dict['A'])
        self.check_keys(dn_cluster['A']['B'].keys(), json_dict['A']['B'])
        self.check_keys(dn_cluster['A']['B']['D'].keys(), json_dict['A']['B']['D'])
        self.check_keys(dn_cluster['A']['C'].keys(), json_dict['A']['C'])

        dn_cluster_dset1 = dn_cluster['A']['B']['dset1']
        json_dict_dset1 = json_dict['A']['B']['dset1']
        self.compare_datasets_json(dn_cluster_dset1, json_dict_dset1)

        dn_cluster_dset2 = dn_cluster['A']['B']['dset2']
        json_dict_dset2 = json_dict['A']['B']['dset2']
        self.compare_datasets_json(dn_cluster_dset2, json_dict_dset2)

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

