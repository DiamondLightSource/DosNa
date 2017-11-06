#!/usr/bin/env python

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
    CONNECTION_CONFIG = {}

    def setUp(self):
        log.info('ClusterTest: %s, %s, %s',
                 self.BACKEND, self.ENGINE, self.CONNECTION_CONFIG)

        dn.use(backend=self.BACKEND, engine=self.ENGINE)
        self.connection_handle = dn.Connection(**self.CONNECTION_CONFIG)

    def tearDown(self):
        self.connection_handle.disconnect()

    def test_config(self):
        self.assertIn(self.BACKEND, dn.backends.available)
        self.assertIn(self.ENGINE, dn.engines.available)

    def test_connection(self):
        connection_handle = dn.Connection(**self.CONNECTION_CONFIG)
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
                        default='test-dosna',
                        help='Connection name')
    parser.add_argument('--connection-options', dest='connection_options',
                        nargs='+', default=[],
                        help='Connection options using the format: '
                             'key1=val1 [key2=val2...]')

    args, unknown_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown_args

    ConnectionTest.BACKEND = args.backend
    ConnectionTest.ENGINE = args.engine
    ConnectionTest.CONNECTION_CONFIG["name"] = args.connection
    ConnectionTest.CONNECTION_CONFIG.update(
        dict(item.split('=') for item in args.connection_options))
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
