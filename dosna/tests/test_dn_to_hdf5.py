import logging
import sys
import unittest

import h5py
import numpy as np
from numpy.testing import assert_array_equal

import dosna as dn
from dosna.backends.base import DatasetNotFoundError
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

H5FILE_NAME = 'testunitfile.h5'
DATASET_NAME = 'fakedataset'


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
        cls.dntohdf = Dosnatohdf5(cls.connection_handle)

    @classmethod
    def tearDownClass(cls):
        cls.connection_handle.disconnect()

    def setUp(self):

        group_A = self.connection_handle.create_group("A", {'A1': 'V1', 'A2': 'V2'})
        group_B = group_A.create_group("B")
        group_C = group_A.create_group("C", {'C1': 'V1'})
        group_D = group_B.create_group("D")

        self.data = np.random.random_integers(DATASET_NUMBER_RANGE[0],
                                              DATASET_NUMBER_RANGE[1],
                                              DATA_SIZE)
        dset1 = group_B.create_dataset('dset1', shape=DATA_SIZE, chunk_size=DATA_CHUNK_SIZE)
        dset2 = group_B.create_dataset('dset2', data=self.data, chunk_size=DATA_CHUNK_SIZE)



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

    def test_same_keys(self):
        dosnadict = self.dntohdf.dosna_to_dict()
        self.dntohdf.dosnadict_to_hdf5(dosnadict, H5FILE_NAME)
        h5_file = h5py.File(H5FILE_NAME)
        self.assertEqual(set(h5_file.keys()), set(self.connection_handle.root_group.keys()))
        self.assertEqual(set(h5_file['A'].keys()), set(self.connection_handle['A'].keys()))
        self.assertEqual(set(h5_file['A/B'].keys()), set(self.connection_handle['A/B'].keys()))
        self.assertEqual(set(h5_file['A/C'].keys()), set(self.connection_handle['A/C'].keys()))
        h5_file.close()

    def test_same_attrs(self):
        dosnadict = self.dntohdf.dosna_to_dict()
        self.dntohdf.dosnadict_to_hdf5(dosnadict, H5FILE_NAME)
        h5_file = h5py.File(H5FILE_NAME)

        dn_attributes = self.connection_handle['A'].attrs
        h5_attributes = dict(attr for attr in h5_file['A'].attrs.items())
        self.assertEqual(dn_attributes, h5_attributes)
        h5_file.close()

    def test_same_dataset_attrs(self):
        dosnadict = self.dntohdf.dosna_to_dict()
        self.dntohdf.dosnadict_to_hdf5(dosnadict, H5FILE_NAME)
        dn_dset = self.connection_handle['A/B'].get_dataset('dset1')
        h5_file = h5py.File(H5FILE_NAME)
        h5_dset = h5_file["/A/B/dset1"]

        self.assertEqual(h5_dset.shape, dn_dset.shape)
        self.assertEqual(h5_dset.dtype, dn_dset.dtype)
        self.assertEqual(h5_dset.chunks, dn_dset.chunk_size)

    def test_same_dataset_chunks(self):
        dn_dset = self.connection_handle["A/B"].get_dataset("dset2")

        dosnadict = self.dntohdf.dosna_to_dict()
        self.dntohdf.dosnadict_to_hdf5(dosnadict, H5FILE_NAME)

        h5_file = h5py.File(H5FILE_NAME)
        h5_dset = h5_file["/A/B/dset2"]

        for chunk in h5_dset.iter_chunks():
            assert_array_equal(dn_dset[chunk], h5_dset[chunk])

    def test_same_dosnadict(self):
        dosnadict = self.dntohdf.dosna_to_dict()
        self.dntohdf.dosnadict_to_hdf5(dosnadict, H5FILE_NAME)

        new_connection = dn.Connection('test_second_connection')
        hdftodn = Hdf5todosna(H5FILE_NAME)
        hdfdict = hdftodn.hdf5_to_dict()
        second_dosnadict = hdftodn.hdf5dict_to_dosna(hdfdict, new_connection)



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

