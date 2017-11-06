

import logging as logging
import sys
import unittest

import dosna as dn
from dosna.tests import configure_logger

log = logging.getLogger(__name__)


class ConnectionTest(unittest.TestCase):
    """
    Test connection handle
    """

    BACKEND = 'ram'
    ENGINE = 'cpu'
    CLUSTER_CONFIG = {}

    def setUp(self):
        log.info('ClusterTest: %s, %s, %s',
                 self.BACKEND, self.ENGINE, self.CLUSTER_CONFIG)

        dn.use(backend=self.BACKEND, engine=self.ENGINE)
        self.connection_handle = dn.Connection(**self.CLUSTER_CONFIG)

    def tearDown(self):
        self.connection_handle.disconnect()

    def test_config(self):
        self.assertIn(self.BACKEND, dn.backends.available)
        self.assertIn(self.ENGINE, dn.engines.available)

    def test_connection(self):
        connection_handle = dn.Connection(**self.CLUSTER_CONFIG)
        self.assertIsNotNone(connection_handle)
        self.assertFalse(connection_handle.connected)
        connection_handle.connect()
        self.assertTrue(connection_handle.connected)
        connection_handle.disconnect()
        self.assertFalse(connection_handle.connected)


def main():
    configure_logger(log)
    import argparse
    parser = argparse.ArgumentParser(description='TestConnection')
    parser.add_argument('--backend', dest='backend', default='ram',
                        help='Select backend (ram | hdf5 | ceph)')
    parser.add_argument('--engine', dest='engine', default='cpu',
                        help='Select engine (cpu | joblib | mpi)')
    parser.add_argument('--connection', dest='connection',
                        default='test-connection',
                        help='Connection name')
    parser.add_argument('--cluster-options', dest='cluster_options', nargs='+',
                        default=[], help='Cluster options using the format: '
                                         'key1=val1 [key2=val2...]')

    args, unknown_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown_args

    ConnectionTest.BACKEND = args.backend
    ConnectionTest.ENGINE = args.engine
    ConnectionTest.CLUSTER_CONFIG["name"] = args.connection
    ConnectionTest.CLUSTER_CONFIG.update(
        dict(item.split('=') for item in args.cluster_options))
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
