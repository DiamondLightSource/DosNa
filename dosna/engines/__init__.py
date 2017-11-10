#!/usr/bin/env python
"""Helper functions to store and get the selected engine"""

from collections import namedtuple
import logging as log

from dosna.util import named_module

AVAILABLE = ['cpu', 'jl', 'mpi']

_current = None

Engine = namedtuple('Engine', ['name', 'Connection', 'Dataset', 'DataChunk',
                               'params'])


def use_engine(engine, **kwargs):
    engine = engine.lower()
    global _current
    if engine in AVAILABLE:
        module_ = named_module('dosna.engines.%s' % engine)
        if hasattr(module_, '_engine'):
            log.debug('Switching engine to `%s`', module_._engine.name)
            _current = module_._engine
            _current.params.update(kwargs)
        else:
            raise Exception('Module `%s` is not a proper engine.' % engine)
    else:
        raise Exception('Engine `{}` not available! Choose one from: {}'
                        .format(engine, AVAILABLE))


def get_engine(name=None):
    if name is not None:
        use_engine(name)
    if _current is None:
        use_engine(AVAILABLE[0])
    return _current
