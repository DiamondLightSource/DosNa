

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
    CONFIG = None

    def setUp(self):
        self.handler = logging.StreamHandler(sys.stdout)
        log.addHandler(self.handler)
        log.info('ClusterTest: {}, {}, {}'
                 .format(self.BACKEND, self.ENGINE, self.CONFIG))

        dn.use(backend=self.BACKEND, engine=self.ENGINE)
        self.C = dn.Cluster(self.CONFIG)

    def tearDown(self):
        log.removeHandler(self.handler)
        self.C.disconnect()

    def test_config(self):
        self.assertIn(self.BACKEND, dn.backends.available)
        self.assertIn(self.ENGINE, dn.engines.available)

    def test_connection(self):
        c = dn.Cluster(self.CONFIG)
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

    args, unknownargs = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknownargs

    ClusterTest.BACKEND = args.backend
    ClusterTest.ENGINE = args.engine
    ClusterTest.CONFIG = args.cluster

    unittest.main(verbosity=2)
