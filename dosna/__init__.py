

import logging as log

from collections import namedtuple

from .base import Backend, Engine
from .backends import use_backend, get_backend
from .engines import use_engine, get_engine


Cluster = Pool = Dataset = DataChunk = None


def use(engine=None, backend=None, engine_kw={}):
    if backend is not None:
        use_backend(backend)
    if engine is not None:
        use_engine(engine, **engine_kw)

    global __current, Cluster, Pool, Dataset, DataChunk
    _, Cluster, Pool, Dataset, DataChunk, _ = __current = get_engine()
    

def compatible(engine, backend):
    if engine.name == 'joblib' and backend.name == 'memory' \
            and engine.params['backend'] == 'multiprocessing':
        return False
    elif engine.name == 'mpi' and backend.name == 'memory':
        return False
    return True


Status = namedtuple('Status', ['engine', 'backend'])
def status(show=False):
    engine = get_engine()
    backend = get_backend()
    if show:
        log.info('---------------------------')
        log.info('Current Engine: %s' % engine.name)
        log.info('Current Backend: %s' % backend.name)
        log.info('---------------------------')
    
    return Status(engine, backend)


use('cpu', 'memory')