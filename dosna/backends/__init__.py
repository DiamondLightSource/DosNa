

import logging as log
from importlib import import_module
from collections import namedtuple


__current = None
__available = ['hdf5']


Backend = namedtuple('Backend', ['name', 'Cluster', 'Pool', 'Dataset', 'DataChunk'])


def use(backend):
    global __current
    if __current is not None:
        log.warn('Cannot setup backend, already set to `%s`',  __current.name)
        return
    if backend in __available:
        m = import_module('dosna.backends.%s' % backend)
        __current = Backend(backend, m.Cluster, m.Pool, m.Dataset, m.DataChunk)
    else:
        raise Exception('Backend `%s` not available!')
        
def backend():
    if __current is None:
        use(__available[0])
    return __current
