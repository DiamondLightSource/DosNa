
import sys
import unittest
import time
import logging
import rados

from dosna.cluster import Cluster


class ClusterTest(unittest.TestCase):

    def setUp(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        self.C = Cluster()
        self.new_pools = []

    def tearDown(self):
        if not self.C.connected:
            return
        for pool in self.new_pools:
            if self.C.has_pool(pool):
                self.log.info('Removing created pool {}'.format(pool))
                self.C.delete_pool(pool)
        self.C.disconnect()

    def test_connection(self):
        c = Cluster().connect()
        self.assertIsNotNone(c)

    def test_connection_error(self):
        self.assertRaises(TypeError, Cluster, conffile=42)
        self.assertRaises(Exception, Cluster, conffile='42')
        self.assertRaises(rados.RadosStateError, self.C.pools)

    def test_check_connection(self):
        self.assertFalse(self.C.connected)
        self.C.connect()
        self.assertTrue(self.C.connected)
        self.C.disconnect()
        self.assertFalse(self.C.connected)

    def test_pool_creation(self):
        self.C.connect()
        dummy_name = 'test_dosna_' + str(int(time.time()))

        self.C.create_pool(dummy_name)
        self.new_pools.append(dummy_name)

        dummy_pool = self.C[dummy_name]
        self.assertIsNotNone(dummy_pool)
        self.assertTrue(self.C.has_pool(dummy_name))
        self.assertIsNotNone(dummy_pool)

        try:
            dummy_pool.require_ioctx_open()
        except rados.IoctxStateError:
            self.fail('Pool {} not opened properly.'.format(dummy_name))

        dummy_pool.close()
        self.assertRaises(rados.IoctxStateError, dummy_pool.require_ioctx_open)
        self.C.delete_pool(dummy_name)
        self.assertFalse(self.C.has_pool(dummy_name))
        self.C.disconnect()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("ClusterTest").setLevel(logging.DEBUG)
    unittest.main()


