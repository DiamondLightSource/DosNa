

import logging as log
from importlib import import_module


__current = None
available = ['cpu']


def use_engine(engine):
    global __current
    if engine in available:
        m = import_module('dosna.engines.%s' % engine)
        if hasattr(m, 'backend'):
            log.debug('Switching engine to `%s`' % m.backend.name)
            __current = m.backend
        else:
            raise Exception('Module `%s` is not a proper engine.' % engine)
    else:
        raise Exception('Engine `{}` not available! Choose one from: {}'
                        .format(engine, available))


def get_engine(name=None):
    if name is not None:
        use_engine(name)
    if __current is None:
        use_engine(available[0])
    return __current