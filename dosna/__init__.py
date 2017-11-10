#!/usr/bin/env python
"""Helper functions to select engine and backend"""

from collections import namedtuple
import logging

from dosna.backends import get_backend, use_backend
from dosna.engines import get_engine, use_engine

log = logging.getLogger(__name__)

_current = Connection = Dataset = DataChunk = None


def use(engine=None, backend=None, engine_kw=None):
    if engine_kw is None:
        engine_kw = {}
    if backend is not None:
        use_backend(backend)
    if engine is not None:
        use_engine(engine, **engine_kw)

    global _current, Connection, Dataset, DataChunk
    _, Connection, Dataset, DataChunk, _ = _current = get_engine()


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
        log.info('Current Engine: %s', engine.name)
        log.info('Current Backend: %s', backend.name)
        log.info('---------------------------')

    return Status(engine, backend)


use('cpu', 'ram')
