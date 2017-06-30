

import sys
import unittest
import logging as logging

import numpy as np
import dosna as dn

logging.basicConfig(level=logging.DEBUG, format="LOG: %(message)s")
log = logging.getLogger()
log.level = logging.INFO


class PoolTest(unittest.TestCase):

    BACKEND = 'ram'
    ENGINE = 'cpu'
    CONFIG = None
    POOL = 'test_dosna'

    def setUp(self):
        self.handler = logging.StreamHandler(sys.stdout)
        log.addHandler(self.handler)
        log.info('PoolTest: {}, {}, {}'
                 .format(self.BACKEND, self.ENGINE, self.CONFIG))

        dn.use(backend=self.BACKEND, engine=self.ENGINE)
        self.C = dn.Cluster(self.CONFIG)
        if self.BACKEND != 'ceph':  # Avoid creating pools in ceph
            self.C.create_pool(self.POOL)
        self.pool = self.C[self.POOL]
        self.fakeds = 'NotADataset'

    def tearDown(self):
        log.removeHandler(self.handler)
        if self.BACKEND != 'ceph':  # Avoid creating pools in ceph
            self.C.del_pool(self.POOL)
        if self.pool.has_dataset(self.fakeds):
            self.pool.del_dataset(self.fakeds)
        self.C.disconnect()

    def test_existing(self):
        self.assertTrue(self.C.has_pool(self.POOL))
        self.assertFalse(self.C.has_pool('NonExistantPool'))

    def test_state(self):
        self.assertFalse(self.pool.isopen)
        self.pool.open()
        self.assertTrue(self.pool.isopen)
        self.pool.close()
        self.assertFalse(self.pool.isopen)
        with self.pool:
            self.assertTrue(self.pool.isopen)
        self.assertFalse(self.pool.isopen)

    def test_dataset_creation(self):
        self.pool.open()
        self.assertFalse(self.pool.has_dataset(self.fakeds))
        self.pool.create_dataset(self.fakeds, data=np.random.rand(10, 10, 10))
        self.assertTrue(self.pool.has_dataset(self.fakeds))
        self.pool.del_dataset(self.fakeds)
        self.pool.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        PoolTest.BACKEND = sys.argv.pop(1)
    if len(sys.argv) > 1:
        PoolTest.ENGINE = sys.argv.pop(1)
    if len(sys.argv) > 1:
        PoolTest.CONFIG = sys.argv.pop(1)
    if len(sys.argv) > 1:
        PoolTest.POOL = sys.argv.pop(1)

    unittest.main(verbosity=2)
