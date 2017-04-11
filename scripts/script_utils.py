

import time


class Timer(object):
    def __init__(self, name='Timer'):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        self.tend = (time.time() - self.tstart)
        print('[%s] Elapsed: %.4f seconds' % (self.name, self.tend))