

import sys
import unittest
import logging as logging

import dosna as dn

logging.basicConfig(level=logging.DEBUG, format="LOG: %(message)s")
log = logging.getLogger()
log.level = logging.INFO


class ClusterTest(unittest.TestCase):

    BACKEND = 'ram'
    ENGINE = 'cpu'
    CLUSTER_CONFIG = {}

    def setUp(self):
        self.handler = logging.StreamHandler(sys.stdout)
        log.addHandler(self.handler)
        log.info('ClusterTest: {}, {}, {}'
                 .format(self.BACKEND, self.ENGINE, self.CLUSTER_CONFIG))

        dn.use(backend=self.BACKEND, engine=self.ENGINE)
        self.C = dn.Cluster(**self.CLUSTER_CONFIG)

    def tearDown(self):
        log.removeHandler(self.handler)
        self.C.disconnect()

    def test_config(self):
        self.assertIn(self.BACKEND, dn.backends.available)
        self.assertIn(self.ENGINE, dn.engines.available)

    def test_connection(self):
        c = dn.Cluster(**self.CLUSTER_CONFIG)
        self.assertIsNotNone(c)
        self.assertFalse(c.connected)
        c.connect()
        self.assertTrue(c.connected)
        c.disconnect()
        self.assertFalse(c.connected)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='TestCluster')
    parser.add_argument('--backend', dest='backend', default='ram',
                        help='Select backend (ram | hdf5 | ceph)')
    parser.add_argument('--engine', dest='engine', default='cpu',
                        help='Select engine (cpu | joblib | mpi)')
    parser.add_argument('--cluster', dest='cluster', default=None,
                        help='Cluster config directory or file '
                        '(backend dependant)')
    parser.add_argument('--cluster-options', dest='cluster_options', nargs='+',
                        default=[], help='Cluster options using the format: '
                                         'key1=val1 [key2=val2...]')

    args, unknownargs = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknownargs

    ClusterTest.BACKEND = args.backend
    ClusterTest.ENGINE = args.engine
    ClusterTest.CLUSTER_CONFIG["name"] = args.cluster
    ClusterTest.CLUSTER_CONFIG.update(
        dict(item.split('=') for item in args.cluster_options))

    unittest.main(verbosity=2)
