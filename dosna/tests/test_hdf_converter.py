import logging
import sys
import unittest
import os
import numpy as np
from numpy.testing import assert_array_equal
import h5py

import dosna as dn
from dosna.tests import configure_logger

from dosna.tools.hdfconverter import HdfConverter

log = logging.getLogger(__name__)

BACKEND = 'ram'
ENGINE = 'cpu'
CONNECTION_CONFIG = {}

DATA_SIZE = (100, 100, 100)
DATA_CHUNK_SIZE = (32, 32, 32)

SEQUENTIAL_TEST_PARTS = 3
DATASET_NUMBER_RANGE = (-100, 100)

H5FILE_NAME = 'test_h5_file.h5'
DATASET_NAME = 'fakedataset'
JSON_FILE_NAME = 'test_json_file.json'
SECOND_H5FILE_NAME = 'test_second_file.h5'

class HdfConverterTest(unittest.TestCase):
    """
    Test HDF5 to DosNa methods
    """
    dn_connection = None

    @classmethod
    def setUpClass(cls):
        dn.use(backend=BACKEND, engine=ENGINE)
        cls.dn_connection = dn.Connection(**CONNECTION_CONFIG)
        cls.dn_connection.connect()
        cls.hdf_converter = HdfConverter()

    @classmethod
    def tearDownClass(cls):
        cls.dn_connection.disconnect()

    def setUp(self):
        def create_h5file(filename):
            with h5py.File(filename, "w") as f:
                A = f.create_group("A")
                B = A.create_group("B")
                C = A.create_group("C")
                D = B.create_group("D")

                A.attrs["A1"] = "V1"
                A.attrs["A2"] = "V2"
                C.attrs["C1"] = "C1"

                # NOT CHUNKED
                dset1 = B.create_dataset("dset1", shape=DATA_SIZE)
                data = np.random.randint(DATASET_NUMBER_RANGE[0], DATASET_NUMBER_RANGE[1]+1, DATA_SIZE)
                dset1[...] = data

                data = np.random.randint(DATASET_NUMBER_RANGE[0], DATASET_NUMBER_RANGE[1] + 1, DATA_SIZE)
                dset2 = B.create_dataset("dset2", shape=DATA_SIZE, chunks=DATA_CHUNK_SIZE)
                dset2[...] = data

                f.close()

        create_h5file(H5FILE_NAME)



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
        if os.path.isfile(SECOND_H5FILE_NAME):
            os.remove(SECOND_H5FILE_NAME)

    def compare_datasets_hdf(self, dataset1, dataset2):
        self.assertEqual(dataset1.name, dataset2.name)
        self.assertEqual(dataset1.shape, dataset2.shape)
        self.assertEqual(dataset1.dtype, dataset2.dtype)
        self.assertEqual(dataset1.chunks, dataset2.chunks)
        self.assertEqual(dataset1.nbytes, dataset2.nbytes)
        self.assertEqual(dataset1.ndim, dataset2.ndim)
        self.assertEqual(dataset1.fillvalue, dataset2.fillvalue)
        for d1, d2 in zip(dataset1, dataset2):
            self.assertIsNone(assert_array_equal(d1, d2))
        if dataset1.chunks is not None:
            self.assertEqual(dataset1.chunks, dataset2.chunks)
            for chunk in dataset1.iter_chunks():
                self.assertIsNone(assert_array_equal(dataset1[chunk], dataset2[chunk]))

    def compare_datasets_dosna(self, hdf_dset, dn_dset):
        self.assertEqual(hdf_dset.shape, dn_dset.shape)
        self.assertEqual(hdf_dset.dtype, dn_dset.dtype)
        self.assertEqual(hdf_dset.ndim, dn_dset.ndim)
        self.assertEqual(hdf_dset.fillvalue, dn_dset.fillvalue)
        for hdf, dn in zip(hdf_dset, dn_dset):
            self.assertIsNone(assert_array_equal(hdf, dn))
        if hdf_dset.chunks is not None:
            self.assertEqual(hdf_dset.chunks, dn_dset.chunk_size)
            for chunk in hdf_dset.iter_chunks():
                self.assertIsNone(assert_array_equal(hdf_dset[chunk], dn_dset[chunk]))

    def compare_datasets_json(self, hdf_dset, json_dset):
        self.assertEqual(hdf_dset.shape, json_dset["shape"])
        self.assertEqual(hdf_dset.dtype, json_dset["dtype"])
        self.assertEqual(hdf_dset.ndim, json_dset["ndim"])
        self.assertEqual(hdf_dset.fillvalue, json_dset["fillvalue"])
        self.assertEqual(hdf_dset.chunks, json_dset["chunk_size"])
        for d1, d2 in zip(hdf_dset, json_dset['dataset_value']):
            self.assertIsNone(assert_array_equal(d1, d2))


    def test_hdf2dict(self):
        hdf_dict = self.hdf_converter.hdf2dict(H5FILE_NAME)
        hdf_file = h5py.File(H5FILE_NAME)
        hdf_file_attrs = dict(attr for attr in hdf_file['A'].attrs.items())
        self.assertDictEqual(hdf_file_attrs, hdf_dict['A']['attrs'])
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['B'].attrs.items())
        self.assertEqual(hdf_file_attrs, hdf_dict['A']['B']['attrs'])
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['B']['D'].attrs.items())
        self.assertEqual(hdf_file_attrs, hdf_dict['A']['B']['D']['attrs'])
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['C'].attrs.items())
        self.assertEqual(hdf_file_attrs, hdf_dict['A']['C']['attrs'])

        hdf_file_dset1 = hdf_file['A']['B']['dset1']
        hdf_dict_dset1 = hdf_dict['A']['B']['dset1']
        self.compare_datasets_hdf(hdf_file_dset1, hdf_dict_dset1)

        hdf_file_dset2 = hdf_file['A']['B']['dset2']
        hdf_dict_dset2 = hdf_dict['A']['B']['dset2']
        self.compare_datasets_hdf(hdf_file_dset2, hdf_dict_dset2)

        hdf_file.close()

    def test_hdf2dosna(self):
        dn_cluster = self.hdf_converter.hdf2dosna(H5FILE_NAME, self.dn_connection)
        hdf_file = h5py.File(H5FILE_NAME)
        hdf_file_attrs = dict(attr for attr in hdf_file['A'].attrs.items())
        self.assertDictEqual(hdf_file_attrs, dn_cluster['A'].attrs)
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['B'].attrs.items())
        self.assertEqual(hdf_file_attrs, dn_cluster['A']['B'].attrs)
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['B']['D'].attrs.items())
        self.assertEqual(hdf_file_attrs, dn_cluster['A']['B']['D'].attrs)
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['C'].attrs.items())
        self.assertEqual(hdf_file_attrs, dn_cluster['A']['C'].attrs)

        hdf_file_dset1 = hdf_file['A']['B']['dset1']
        dn_cluster_dset1 = dn_cluster['A']['B']['dset1']
        self.compare_datasets_dosna(hdf_file_dset1, dn_cluster_dset1)

        hdf_file_dset2 = hdf_file['A']['B']['dset2']
        dn_cluster_dset2 = dn_cluster['A']['B']['dset2']
        self.compare_datasets_dosna(hdf_file_dset2, dn_cluster_dset2)

        hdf_file.close()

    def test_hdf2json(self):
        json_dict = self.hdf_converter.hdf2json(H5FILE_NAME, JSON_FILE_NAME)
        hdf_file = h5py.File(H5FILE_NAME)
        hdf_file_attrs = dict(attr for attr in hdf_file['A'].attrs.items())
        self.assertDictEqual(hdf_file_attrs, json_dict['A']['attrs'])
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['B'].attrs.items())
        self.assertEqual(hdf_file_attrs, json_dict['A']['B']['attrs'])
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['B']['D'].attrs.items())
        self.assertEqual(hdf_file_attrs, json_dict['A']['B']['D']['attrs'])
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['C'].attrs.items())
        self.assertEqual(hdf_file_attrs, json_dict['A']['C']['attrs'])

        hdf_file_dset1 = hdf_file['A']['B']['dset1']
        json_dict_dset1 = json_dict['A']['B']['dset1']
        self.compare_datasets_json(hdf_file_dset1, json_dict_dset1)

        hdf_file_dset2 = hdf_file['A']['B']['dset2']
        json_dict_dset2 = json_dict['A']['B']['dset2']
        self.compare_datasets_json(hdf_file_dset2, json_dict_dset2)

        hdf_file.close()

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
