

import logging as log
from importlib import import_module


__current = None
available = ['mem', 'hdf5']


def use_backend(backend):
    global __current
    if backend in available:
        m = import_module('dosna.backends.%s' % backend)
        if hasattr(m, 'backend'):
            log.debug('Switching backend to `%s`' % m.backend.name)
            __current = m.backend
        else:
            raise Exception('Module `%s` is not a proper backend.' % backend)
    else:
        raise Exception('Backend `{}` not available! Choose from: {}'
                        .format(backend, available))


def get_backend(name=None):
    if name is not None:
        use_backend(name)
    if __current is None:
        use_backend(available[0])
    return __current
