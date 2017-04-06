

import numpy as np


def shape2str(t, sep='::'):
    return sep.join(map(str, t))

def str2shape(s, sep='::'):
    return tuple(map(int, s.split(sep)))

def dtype2str(dtype):
    return np.dtype(dtype).str