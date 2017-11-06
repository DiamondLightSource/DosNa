#!/usr/bin/env python
"""
Helper functions to save the selected backend
"""

import logging
from importlib import import_module

log = logging.getLogger(__name__)

_current = None
available = ['ram', 'hdf5', 'ceph']


def use_backend(backend):
    backend = backend.lower()
    global _current
    if backend in available:
        module = import_module('dosna.backends.{}'.format(backend))
        if hasattr(module, '__backend__'):
            log.debug('Switching backend to `%s`', module.__backend__.name)
            _current = module.__backend__
        else:
            raise Exception(
                'Module `{}` is not a proper backend.'.format(backend))
    else:
        raise Exception('Backend `{}` not available! Choose from: {}'
                        .format(backend, available))


def get_backend(name=None):
    if name is not None:
        use_backend(name)
    if _current is None:
        use_backend(available[0])
    return _current
