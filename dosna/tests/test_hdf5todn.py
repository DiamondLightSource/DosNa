import logging
import sys
import unittest

import numpy as np
import h5py

import dosna as dn
from dosna.backends.base import DatasetNotFoundError
from dosna.tests import configure_logger

from dosna.tools import hdf5todosna

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
        
        cls.hdftodn = hdf5todosna.Hdf5todosna(H5FILE_NAME) #TODO double?
        
    @classmethod
    def tearDownClass(cls):
        cls.connection_handle.disconnect()
        
    def setUp(self):
        self.fake_dataset = 'NotADataset'
        self.data = np.random.random_integers(DATASET_NUMBER_RANGE[0],
                                              DATASET_NUMBER_RANGE[1],
                                              DATA_SIZE)
        """
        with h5py.File(H5FILE_NAME, "w") as f:
            self.dataset = f.create_dataset(
                self.fake_dataset, data=self.data, chunks=DATA_CHUNK_SIZE
            )


    def tearDown(self):
        with h5py.File(H5FILE_NAME, "a") as f:
            del f[self.fake_dataset]

    def test_existing_dataset(self):
        with h5py.File(H5FILE_NAME, "r") as f:
            dataset = f[self.fake_dataset]
            self.assertIsInstance(dataset, h5py._hl.dataset.Dataset)
            
    def test_number_chunks(self):
        self.assertSequenceEqual(list(self.dataset.chunks), [32, 32, 32])
    
    def test_hdf5dataset_equals_hdf5dictdataset(self):
        with h5py.File(H5FILE_NAME, "r") as f:
            dataset = f[self.fake_dataset]
    """
    def test_hdf5_to_dosna(self):
        pass
    
    def test_hdf5dict_to_json(self):
        pass
    
    def test_jsondict_to_jsonfile(self):
        pass
    
    def test_jsonfile_to_jsondict(self):
        pass
    
    def test_jsondict_to_dosna(self):
        pass
    
    def test_existing_dataset_dosna(self):
        self.hdf5dict = self.hdftodn.hdf5_to_dict()
        #self.assertEqual(dataset, self.hdf5dict[self.fake_dataset])
        #self.hdftodn.hdf5dict_to_dosna()
        #print(self.connection_handle.has_dataset(self.fake_dataset))
        pass
    
    def test_hdf5dataset_equals_dosnadataset(self):
        """
        Test values of HDF5 dataset are equal to the values of DosNa dataset
        """
        pass
    
    def test_hdf5dict_equals_dosnadict(self):
        """
        Test dictionary of HDF5 objects equals the dictionary of DosNa objects
        with the objects being the only difference
        """
        pass
        
    
        
        
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
        
