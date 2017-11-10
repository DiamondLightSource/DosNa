#!/usr/bin/env python
"""hard to classify useful functions"""

from __future__ import print_function

import logging
import time
from os.path import join

import numpy as np

log = logging.getLogger(__name__)


def shape2str(dims, sep='::'):
    return sep.join(map(str, dims))


def str2shape(string, sep='::'):
    return tuple(map(int, string.split(sep)))


def dtype2str(dtype):
    return np.dtype(dtype).str


class Timer(object):
    def __init__(self, name='Timer'):
        self.name = name
        self.tstart = -1
        self.tend = -1
        self.time = 0

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, *args):
        self.tend = time.time()
        self.time = self.tend - self.tstart
        print('[*] %s -- Elapsed: %.4f seconds'.format(self.name, self.time))


class DirectoryTreeMixin(object):

    @property
    def path(self):
        if getattr(self, "directory", False):
            return join(self.directory, self.name)
        return self.parent.relpath(self.name)

    def relpath(self, name=''):
        return join(self.path, name)


def named_module(name):
    # convenient function to unify all named imports
    from importlib import import_module
    return import_module(name)
