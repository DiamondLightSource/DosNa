

import sys
import unittest
import logging

import numpy as np

import dosna as dn
dn.auto_init()


class AutoInitTest(unittest.TestCase):

    def setUp(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.DEBUG)

    def test_cluster_is_connected(self):
        with dn.Pool('test_dosna_autoinit') as P:
            P.create_dataset('AutoInit', data=np.random.randn(100, 100, 100))
            P.delete()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("AutoInitTest").setLevel(logging.DEBUG)
    unittest.main()