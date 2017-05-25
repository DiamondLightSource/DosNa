

import time
import numpy as np


def shape2str(t, sep='::'):
    return sep.join(map(str, t))


def str2shape(s, sep='::'):
    return tuple(map(int, s.split(sep)))


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

    def __exit__(self, type, value, traceback):
        self.tend = time.time()
        self.time = self.tend - self.tstart
        print('[*] %s -- Elapsed: %.4f seconds' % (self.name, self.time))