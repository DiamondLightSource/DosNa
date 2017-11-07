#!/usr/bin/env python
"""Helper functions used in dosna tests"""

import logging
import sys


def configure_logger(log):
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    log.addHandler(stream_handler)
    log.setLevel(logging.DEBUG)
