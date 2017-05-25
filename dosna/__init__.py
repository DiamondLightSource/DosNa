

import logging as log

from .base import Backend, Engine
from .backends import use_backend, get_backend
from .engines import use_engine, get_engine


Cluster = Pool = Dataset = DataChunk = None


def use(engine=None, backend=None):
    if backend is not None:
        use_backend(backend)
    if engine is not None:
        use_engine(engine)

    global __current, Cluster, Pool, Dataset, DataChunk
    _, Cluster, Pool, Dataset, DataChunk = __current = get_engine()
    

def status(show=False):
    engine = get_engine()
    backend = get_backend()
    if show:
        log.info('---------------------------')
        log.info('Current Engine: %s' % engine.name)
        log.info('Current Backend: %s' % backend.name)
        log.info('---------------------------')
    return engine, backend
    

use('cpu', 'mem')