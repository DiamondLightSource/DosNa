#!/usr/bin/env python

import logging
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
    connection_handle = None

    @classmethod
    def setUpClass(cls):
        log.info('ConnectionTest: %s, %s, %s',
                 cls.BACKEND, cls.ENGINE, cls.CONNECTION_CONFIG)

        dn.use(backend=cls.BACKEND, engine=cls.ENGINE)
        cls.connection_handle = dn.Connection(**cls.CONNECTION_CONFIG)

    def test_config(self):
        self.assertIn(self.BACKEND, dn.backends.AVAILABLE)
        self.assertIn(self.ENGINE, dn.engines.AVAILABLE)

    def test_connection(self):
        self.assertIsNotNone(self.connection_handle)
        self.assertFalse(self.connection_handle.connected)
        self.connection_handle.connect()
        self.assertTrue(self.connection_handle.connected)
        self.connection_handle.disconnect()
        self.assertFalse(self.connection_handle.connected)


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
