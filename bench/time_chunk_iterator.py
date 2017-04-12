

import sys
import os.path as op
import time

import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NTEST = 3
CHUNKS = (128, 256, 512, 1024)

parent_folder = op.realpath(op.join(op.dirname(__file__), '..'))
results_folder = op.join(op.dirname(__file__), 'results')
sys.path.append(parent_folder)

import dosna as dn

with dn.Cluster(conffile=op.join(parent_folder, 'ceph.conf')) as C:
    pool = C.create_pool(test=True)
    pool.write('dummy', 'dummy')

    print('Creating sample data')
    timings = np.zeros((len(CHUNKS), NTEST, 5), np.float32)

    for i, chunk_size in enumerate(CHUNKS):
        print "CHUNKS", chunk_size
        for j in range(NTEST):
            print('Dumping data onto memory')
            t0 = time.time()
            dsdata = pool.create_dataset('data', shape=(1024, 1024, 1024),
                                         dtype=np.float32, chunks=chunk_size)
            t1 = time.time()
            timings[i, j, 0] = t1 - t0

            print('Process slices')
            t0 = time.time()
            slices = dsdata._process_slices((slice(None), slice(None), slice(None)))
            t1 = time.time()
            timings[i, j, 1] = t1 - t0

            print('Get Chunk Iterator')
            t0 = time.time()
            it = dsdata._get_chunk_slice_iterator(slices)
            t1 = time.time()
            timings[i, j, 2] = t1 - t0

            print('Consume Chunk Iterator')
            t0 = time.time()
            for x in it():
                x;
            t1 = time.time()
            timings[i, j, 3] = t1 - t0

            print('Deleting dataset')
            t0 = time.time()
            dsdata.delete()
            t1 = time.time()
            timings[i, j, 4] = t1 - t0

    pool.close()
    pool.delete()

t = int(time.time())

mean_times = timings.mean(axis=1)
std_times = timings.std(axis=1)

labels = ['Create', 'Slices', 'Chunks', 'Consume', 'Delete']
inds = np.arange(len(labels))
margin = 0.05
width = (1. - inds.size * margin) / len(CHUNKS)

fig, ax = plt.subplots()
for i in range(len(CHUNKS)):
    ax.bar(inds + i * width + margin, mean_times[i],
           width=width, yerr=std_times[i], label=CHUNKS[i])

ax.set_ylabel('Seconds')
ax.set_xticks(inds + len(CHUNKS) * width / 2)
ax.set_xticklabels(labels)
plt.legend()
plt.savefig(op.join(results_folder, 'timings_chunk-iterator_{}.png'.format(t)))

CHUNKS = CHUNKS[-2:]
mean_times = mean_times[-2:]
std_times = std_times[-2:]

inds = np.arange(len(labels))
margin = 0.05
width = (1. - inds.size * margin) / len(CHUNKS)

fig, ax = plt.subplots()
for i in range(len(CHUNKS)):
    ax.bar(inds + i * width + margin, mean_times[i],
           width=width, yerr=std_times[i], label=CHUNKS[i])

ax.set_ylabel('Seconds')
ax.set_xticks(inds + len(CHUNKS) * width / 2)
ax.set_xticklabels(labels)
plt.legend()
plt.savefig(op.join(results_folder, 'timings_chunk-iterator_{}_2.png'.format(t)))
