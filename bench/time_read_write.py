

import sys
import os.path as op
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NTEST = 3
CHUNKS = (128, 256,)

parent_folder = op.realpath(op.join(op.dirname(__file__), '..'))
results_folder = op.join(op.dirname(__file__), 'results')
sys.path.append(parent_folder)

import dosna as dn

with dn.Cluster(conffile=op.join(parent_folder, 'ceph.conf')) as C:
    pool = C.create_pool(test=True)
    pool.write('dummy', 'dummy')

    print('Creating sample data')
    npdata = np.random.randn(512, 512, 512).astype(np.float32)

    timings = np.zeros((len(CHUNKS), NTEST, 3), np.float32)

    for i, chunk_size in enumerate(CHUNKS):
        print "CHUNKS", chunk_size
        for j in range(NTEST):
            print('Dumping data onto memory')
            t0 = time.time()
            dsdata = pool.create_dataset('data', data=npdata, chunks=chunk_size)
            t1 = time.time()
            timings[i, j, 0] = t1 - t0

            print('Reading all the data')
            t0 = time.time()
            dsdata[:]
            t1 = time.time()
            timings[i, j, 1] = t1 - t0

            print('Deleting dataset')
            t0 = time.time()
            dsdata.delete()
            t1 = time.time()
            timings[i, j, 2] = t1 - t0

    pool.close()
    pool.delete()

import h5py as h5
import os

if not os.path.isdir('/tmp/to_remove'):
    os.mkdir('/tmp/to_remove')
Htimings = np.zeros((len(CHUNKS), NTEST, 3), np.float32)

for i, chunk_size in enumerate(CHUNKS):
    print "CHUNKS", chunk_size
    for j in range(NTEST):
        with h5.File('/tmp/to_remove/tmp.h5') as f:
            print('Dumping data onto memory')
            t0 = time.time()
            dsdata = f.create_dataset('data', data=npdata, chunks=(chunk_size, chunk_size, chunk_size))
            t1 = time.time()
            Htimings[i, j, 0] = t1 - t0

            print('Reading all the data')
            t0 = time.time()
            dsdata[:]
            t1 = time.time()
            Htimings[i, j, 1] = t1 - t0

        print('Deleting dataset')
        t0 = time.time()
        os.remove('/tmp/to_remove/tmp.h5')
        t1 = time.time()
        Htimings[i, j, 2] = t1 - t0

mean_times = timings.mean(axis=1)
std_times = timings.std(axis=1)

labels = ['Write', 'Read', 'Delete']
inds = np.arange(len(labels))

plt.figure()

plt.subplot(1, 2, 1)
for i in range(len(CHUNKS)):
    plt.errorbar(inds, mean_times[i], yerr=std_times[i], label=CHUNKS[i])
plt.ylabel('Seconds')
plt.ylim([timings.max(), 0])
plt.xticks(range(len(labels)), labels)
plt.legend()

mean_times = Htimings.mean(axis=1)
std_times = Htimings.std(axis=1)

plt.subplot(1, 2, 2)
for i in range(len(CHUNKS)):
    plt.errorbar(inds, mean_times[i], yerr=std_times[i], label=CHUNKS[i])
plt.ylabel('Seconds')
plt.ylim([Htimings.max(), 0])
plt.xticks(range(len(labels)), labels)
plt.legend()

plt.savefig(op.join(results_folder, 'timings_read-write_{}.png'.format(int(time.time()))))
