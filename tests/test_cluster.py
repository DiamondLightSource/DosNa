

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
    if len(sys.argv) > 1:
        ClusterTest.BACKEND = sys.argv.pop(1)
    if len(sys.argv) > 1:
        ClusterTest.ENGINE = sys.argv.pop(1)
    if len(sys.argv) > 1:
        ClusterTest.CONFIG = sys.argv.pop(1)

    unittest.main(verbosity=2)
