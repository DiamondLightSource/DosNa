import logging
import sys
import unittest

import numpy as np
from numpy.testing import assert_array_equal
import h5py

import dosna as dn
from dosna.backends.base import DatasetNotFoundError, GroupNotFoundError
from dosna.tests import configure_logger

from dosna.tools.dosnatohdf5 import Dosnatohdf5
from dosna.tools.hdf5todosna import Hdf5todosna

log = logging.getLogger(__name__)

BACKEND = 'ram'
ENGINE = 'cpu'
CONNECTION_CONFIG = {}

DATA_SIZE = (100, 100, 100)
DATA_CHUNK_SIZE = (32, 32, 32)

SEQUENTIAL_TEST_PARTS = 3
DATASET_NUMBER_RANGE = (-10000, 10000)

H5FILE_NAME = 'testh5file.h5'
DATASET_NAME = 'fakedataset'
JSON_FILE_NAME = 'testjsonfile.json'


class Hdf5todosnaTest(unittest.TestCase):
    """
    Test HDF5 to DosNa methods
    """
    connection_handle = None

    @classmethod
    def setUpClass(cls):
        dn.use(backend=BACKEND, engine=ENGINE)
        cls.connection_handle = dn.Connection(**CONNECTION_CONFIG)
        cls.connection_handle.connect()
        cls.hdftodn = Hdf5todosna(H5FILE_NAME)

    @classmethod
    def tearDownClass(cls):
        cls.connection_handle.disconnect()

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

                #NOT CHUNKED
                dset1 = B.create_dataset("dset1", shape=DATA_SIZE)
                data = np.random.random_integers(DATASET_NUMBER_RANGE[0],
                                                 DATASET_NUMBER_RANGE[1],
                                                 DATA_SIZE)
                dset1[...] = data
                data = np.random.random_integers(DATASET_NUMBER_RANGE[0],
                                                 DATASET_NUMBER_RANGE[1],
                                                 DATA_SIZE)
                dset2 = B.create_dataset("dset2", shape=DATA_SIZE, chunks=DATA_CHUNK_SIZE)
                dset2[...] = data
                f.close()

        create_h5file(H5FILE_NAME)


    def tearDown(self):
        group_A = self.connection_handle.get_group('A')
        group_B = group_A.get_group('B')
        group_C = group_A.get_group('C')
        group_D = group_B.get_group('D')

        group_B.del_group('D')
        group_A.del_group('C')
        group_A.del_group('B')
        self.connection_handle.del_group('A')
        self.connection_handle.disconnect()

    def test_hdf5_equals_dosnadict(self):
        hdf5dict = self.hdftodn.hdf5_to_dict()
        dosnadict = self.hdftodn.hdf5dict_to_dosna(hdf5dict, self.connection_handle)
        self.assertEqual(set(hdf5dict), set(dosnadict))
        self.assertEqual(set(hdf5dict['A']), set(dosnadict['A']))
        self.assertEqual(set(hdf5dict['A']['B']), set(dosnadict['A']['B']))


    def test_hdf5_equals_dosna(self):
        hdf5dict = self.hdftodn.hdf5_to_dict()
        self.hdftodn.hdf5dict_to_dosna(hdf5dict, self.connection_handle)

        h5_file = h5py.File(H5FILE_NAME)
        self.assertEqual(set(h5_file.keys()), set(self.connection_handle.root_group.keys()))
        self.assertEqual(set(h5_file['A'].keys()), set(self.connection_handle['A'].keys()))
        #self.assertCountEqual(list(h5_file['A/B'].keys()), self.connection_handle['A/B'].keys())
        self.assertEqual(set(h5_file['A/C'].keys()), set(self.connection_handle['A/C'].keys()))
        h5_file.close()

    def test_same_attrs(self):
        hdf5dict = self.hdftodn.hdf5_to_dict()
        self.hdftodn.hdf5dict_to_dosna(hdf5dict, self.connection_handle)

        h5_file = h5py.File(H5FILE_NAME)

        dn_attributes = self.connection_handle['A'].attrs
        h5_attributes = dict(attr for attr in h5_file['A'].attrs.items())
        self.assertEqual(dn_attributes, h5_attributes)
        h5_file.close()

    def test_same_dataset_attrs(self):
        hdf5dict = self.hdftodn.hdf5_to_dict()
        self.hdftodn.hdf5dict_to_dosna(hdf5dict, self.connection_handle)

        dn_dset_name = ''.join(k for k in self.connection_handle["A/B"].keys() if k.startswith("dset1"))
        dn_dset = self.connection_handle["A/B"].get_dataset(dn_dset_name)

        h5_file = h5py.File(H5FILE_NAME)
        h5_dset = h5_file["/A/B/dset1"]

        self.assertEqual(h5_dset.shape, dn_dset.shape)
        self.assertEqual(h5_dset.dtype, dn_dset.dtype)
        #self.assertEqual(h5_dset.chunks, dn_dset.chunk_size)

        h5_file.close()

    def test_same_datachunks(self):
        hdf5dict = self.hdftodn.hdf5_to_dict()
        self.hdftodn.hdf5dict_to_dosna(hdf5dict, self.connection_handle)
        dn_dset_name = ''.join(k for k in self.connection_handle["A/B"].keys() if k.startswith("dset2"))
        dn_dset = self.connection_handle["A/B"].get_dataset(dn_dset_name)

        h5_file = h5py.File(H5FILE_NAME)
        h5_dset = h5_file["/A/B/dset2"]

        for chunk in h5_dset.iter_chunks():
            assert_array_equal(dn_dset[chunk], h5_dset[chunk])
        h5_file.close()

    def test_same_dosnadicts(self):
        hdf5dict = self.hdftodn.hdf5_to_dict()
        dosnadict = self.hdftodn.hdf5dict_to_dosna(hdf5dict, self.connection_handle)
        jsondict = self.hdftodn.hdf5dict_to_json(hdf5dict, JSON_FILE_NAME)
        second_dosnadict = self.hdftodn.json_to_dosna(JSON_FILE_NAME, self.connection_handle)
        self.assertEqual(set(dosnadict.keys()), set(second_dosnadict.keys()))
        self.assertEqual(set(dosnadict['A'].keys()), set(second_dosnadict['A'].keys()))
        self.assertEqual(set(dosnadict['A']['B'].keys()), set(second_dosnadict['A']['B'].keys()))

    def test_same_hdf5(self):
        hdf5dict = self.hdftodn.hdf5_to_dict()
        dosnadict = self.hdftodn.hdf5dict_to_dosna(hdf5dict, self.connection_handle)

        h5_filename = 'test_second_file.h5'
        dntohdf = Dosnatohdf5(self.connection_handle)
        second_dosnadict = dntohdf.dosna_to_dict()
        dntohdf.dosnadict_to_hdf5(second_dosnadict, h5_filename)
        second_hdf5dict = dntohdf.hdf5file_to_hdf5dict(h5_filename)

        # TODO assert


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

