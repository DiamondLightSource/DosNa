

import time
import h5py as h5
import numpy as np
import os.path as op

import dosna.mpi as dn

import matplotlib

try:
    matplotlib.use('Qt4Agg')
except:
    matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

###############################################################################

# Init dosna
dn.auto_init()
# Chunk shapes
CSIZE = (24, 32, 48, 64, 72, 80, 96, 112, 128)
# Number of tests for timing estimates
NTEST = 10

###############################################################################
# Data

path = op.realpath(op.join(op.dirname(__file__), '..', 'test.h5'))
h5f = h5.File(path, 'r')
data = h5f['data']
dn.pprint(data.shape)

###############################################################################
# Load Data

pool = dn.Pool()

times =[]

for csize in CSIZE:
    dn.pprint('Runing tests for Chunk Size: {}'.format(csize), rank=0)
    total = 0
    for _ in range(NTEST):
        dn.wait()
        t0 = time.time()

        ds = pool.create_dataset('data', data=data, chunks=csize)

        dn.wait()
        t1 = time.time()
        total += (t1 - t0)
        ds.delete()

    average = total / NTEST
    dn.pprint('Average time for {0} tests with chunk size {1}: {2:.4f}'
              .format(NTEST, csize, average), rank=0)
    times.append(average)
    dn.wait()

dn.wait()

###############################################################################
# Plot

x = range(len(CSIZE))
y = times

#plt.figure()
#plt.title('{} with {} cores'.format(data.shape, dn.ncores))
#plt.plot(x, y)
#plt.show()

dn.pprint(CSIZE, rank=0)
dn.pprint(times, rank=0)

###############################################################################
# End script

dn.wait()

h5f.close()
pool.close()
pool.delete()
