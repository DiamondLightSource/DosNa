#!/usr/bin/env python
import logging
import os
import unittest

from pyclovis import Clovis

log = logging.getLogger(__name__)

TEST_OBJECT_NAME = "test-object-102"
TEST_METADATA = {"length": 20, "name": TEST_OBJECT_NAME}
TEST_DATA_LEN = 100
TEST_CHUNK_ID = 0
TEST_CONFIG_FILE = "pyclovis_test.conf"


class PyclovisTest(unittest.TestCase):
    clovis = None

    @classmethod
    def setUpClass(cls):
        cls.clovis = Clovis(conffile=TEST_CONFIG_FILE)
        log.info("Connecting ...\n")
        cls.clovis.connect()
        log.info("Creating object metadata for %s\n", TEST_OBJECT_NAME)
        cls.clovis.create_object_metadata(TEST_OBJECT_NAME)

    @classmethod
    def tearDownClass(cls):
        log.info("Deleting object metadata for %s\n", TEST_OBJECT_NAME)
        cls.clovis.delete_object_metadata(TEST_OBJECT_NAME)
        cls.clovis.disconnect()

    def setUp(self):
        log.info("Creating object chunk %d\n", TEST_CHUNK_ID)
        self.clovis.create_object_chunk(TEST_OBJECT_NAME, TEST_CHUNK_ID)

    def tearDown(self):
        log.info("Deleting object chunk %d\n", TEST_CHUNK_ID)
        self.clovis.delete_object_chunk(TEST_OBJECT_NAME, TEST_CHUNK_ID)

    def test_is_connected(self):
        self.assertTrue(self.clovis.connected)

    def test_have_object(self):
        self.assertTrue(self.clovis.has_object_chunk(TEST_OBJECT_NAME,
                                                     TEST_CHUNK_ID))

    def test_dont_have_object(self):
        self.assertFalse(self.clovis.has_object_chunk("NONEXISTENT",
                                                      TEST_CHUNK_ID))

    def test_set_and_get_metadata(self):
        self.clovis.set_object_metadata(TEST_OBJECT_NAME, TEST_METADATA)
        stored_data = self.clovis.get_object_metadata(TEST_OBJECT_NAME)
        self.assertEqual(TEST_METADATA, stored_data)

    def test_write_and_read_random_chunk(self):
        test_data = os.urandom(TEST_DATA_LEN)
        self.clovis.write_object_chunk(TEST_OBJECT_NAME, TEST_CHUNK_ID,
                                       test_data, TEST_DATA_LEN)
        stored_data = self.clovis.read_object_chunk(TEST_OBJECT_NAME,
                                                    TEST_CHUNK_ID,
                                                    TEST_DATA_LEN)
        self.assertEqual(test_data, stored_data)

    def _get_sample_bytes(self, n_bytes):
        return b"".join([chr(i % 26 + 65) for i in range(n_bytes)])

    def test_write_and_read_max_chunk(self):
        max_len = self.clovis.get_option("block_size")
        test_data = self._get_sample_bytes(max_len)
        self.clovis.write_object_chunk(TEST_OBJECT_NAME, TEST_CHUNK_ID,
                                       test_data, max_len)
        stored_data = self.clovis.read_object_chunk(TEST_OBJECT_NAME,
                                                    TEST_CHUNK_ID, max_len)
        self.assertEqual(test_data, stored_data)

    def test_write_and_read_more_chunks(self):
        max_len = self.clovis.get_option("block_size")
        sample_len = max_len * 2 + max_len // 2
        test_data = self._get_sample_bytes(sample_len)
        self.clovis.write_object_chunk(TEST_OBJECT_NAME, TEST_CHUNK_ID,
                                       test_data, sample_len)
        stored_data = self.clovis.read_object_chunk(TEST_OBJECT_NAME,
                                                    TEST_CHUNK_ID, sample_len)
        self.assertEqual(len(test_data), len(stored_data))
        self.assertEqual(test_data, stored_data)


def main():
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=5)


if __name__ == "__main__":
    main()
