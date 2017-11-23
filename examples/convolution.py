#!/usr/bin/env python
"""3D convolution using a gaussian filter

All the data is managed using dosna"""

from __future__ import print_function

import argparse
import tempfile
from os.path import join

import h5py as h5
import numpy as np

try:
    from scipy.misc import imsave
except ImportError:
    pass

from scipy.ndimage import gaussian_filter, gaussian_filter1d

import dosna as dn
from dosna.util.mpi import (MpiTimer, mpi_barrier, mpi_comm, mpi_rank,
                            mpi_root, mpi_size, pprint)

###############################################################################
# Test parameters

parser = argparse.ArgumentParser(description='Test Gaussian Convolution')
parser.add_argument('--backend', dest='backend', default='hdf5',
                    help='Select backend to use (ram | *hdf5 | ceph)')
parser.add_argument('--engine', dest='engine', default='mpi',
                    help='Select engine to use (cpu | joblib | *mpi)')
parser.add_argument('--connection', dest='connection', default='test-dosna',
                    help='Connection name')
parser.add_argument('--connection-options', dest='connection_options',
                    nargs='+', default=[],
                    help='Cluster options using the format: '
                         'key1=val1 [key2=val2...]')
parser.add_argument('--out', dest='out', default='.',
                    help='Output directory for the results (default ".").')

parser.add_argument('--data-sizes', dest='data_sizes', type=int, nargs='+',
                    default=[128, 256, 512],
                    help='List of sizes for datasets to test in. '
                    'Sizes are 3D, e.g. 128 -> 128x128x128')
parser.add_argument('--chunk-sizes', dest='chunk_sizes', type=int, nargs='+',
                    default=[24, 36, 48, 60, 72, 96, 128],
                    help='List of sizes for chunking the datasets. '
                    'Sizes are 3D, e.g. 32 -> 32x32x32')

parser.add_argument('--sigma', dest='sigma', type=float, default=1.5,
                    help='Determines the sigma to be used for the '
                    'Gaussian Convolution')

parser.add_argument('--ntest', dest='ntest', type=int, default=10,
                    help='Number of tests to be run for each data size '
                    'and chunk size')
parser.add_argument('--tmp', dest='tmp', type=str, default=None,
                    help='Temporary directory to store the initial dataset, '
                         'make sure it is visible to every process if you are '
                         'using mpi')

args = parser.parse_args()

###############################################################################

DATA_SIZE = args.data_sizes
CHUNK_SIZE = args.chunk_sizes

dn.use(backend=args.backend, engine=args.engine)

NTESTS = args.ntest
SIGMA = args.sigma
TRUNC = 3
CONNECTION_CONFIG = {"name": args.connection}
CONNECTION_CONFIG.update(
    dict(item.split('=') for item in args.connection_options))
OUT_PATH = args.out
TMP=args.tmp

engine, backend = dn.status()
pprint('Starting Test == Backend: {}, Engine: {}, config: {}, Out: {}'
       .format(backend.name, engine.name, CONNECTION_CONFIG, OUT_PATH), rank=0)


###############################################################################
# Load data

def create_random_dataset(DS):
    pprint('Populating random data on disk', rank=0)

    if mpi_root():
        f = tempfile.NamedTemporaryFile(dir=TMP)
        data = np.random.rand(DS, DS, DS).astype(np.float32)
        fname = f.name + '.h5'
        with h5.File(fname, 'w') as g:
            g.create_dataset('data', data=data)
    else:
        fname = None
    fname = mpi_comm().bcast(fname, root=0)
    pprint(fname)

    h5in = h5.File(fname, 'r')
    h5data = h5in['data']
    return h5in, h5data


###############################################################################
# Output Dataset

def get_output_dataset():
    # only one node should create results
    dname = '{}/{}'.format(backend.name, mpi_size())
    fname = join(OUT_PATH, 'result.h5')
    shape = (len(DATA_SIZE), len(CHUNK_SIZE), 3, NTESTS)

    with h5.File(fname, 'a') as g:
        if dname in g:
            if g[dname].shape != shape:
                raise ValueError('Dataset exists with invalid shape')
        else:
            g.create_dataset(dname, shape=shape, dtype=np.float32)
            g[dname].attrs['axis0_label'] = 'Data Size'
            g[dname].attrs['axis1_label'] = 'Chunk Size'
            g[dname].attrs['axis2_label'] = 'Tests'
            g[dname].attrs['axis3_label'] = '# Tests'
            g[dname].attrs['axis0_value'] = DATA_SIZE
            g[dname].attrs['axis1_value'] = CHUNK_SIZE
            g[dname].attrs['axis2_value'] = np.array(['DATA', '3D', '1D'], dtype='S4')
            g[dname].attrs['axis3_value'] = list(range(NTESTS))

    h5out = h5.File(fname, 'a')
    h5result = h5out[dname]
    return h5out, h5result


###############################################################################
# Convolution 1

def convolve1(ds, sigma):
    mpi_barrier()

    pprint('Convolving in 3D', rank=0)

    with MpiTimer('Separable 3D convolution') as T:
        ds3d_ = ds.clone('gaussian_3d')

        for z in range(mpi_rank(), ds.chunk_grid[0], mpi_size()):
            zS = z * ds.chunk_size[0]
            zE = min(zS + ds.chunk_size[0], ds.shape[0])
            for y in range(ds.chunk_grid[1]):
                yS = y * ds.chunk_size[1]
                yE = min(yS + ds.chunk_size[1], ds.shape[1])
                for x in range(ds.chunk_grid[2]):
                    xS = x * ds.chunk_size[2]
                    xE = min(xS + ds.chunk_size[2], ds.shape[2])
                    out = gaussian_filter(ds[zS:zE, yS:yE, xS:xE], sigma=sigma)
                    ds3d_[zS:zE, yS:yE, xS:xE] = out
        mpi_barrier()

    if mpi_root():
        try:
            imsave(join(OUT_PATH, 'conv3d_{}-{}-{}.png'.format(
                   mpi_size(), ds.shape[0], ds.chunk_size[0])),
                   ds3d_[ds.shape[0] // 2])
        except NameError:
            pass
    mpi_barrier()

    ds3d_.delete()

    return T.time


###############################################################################
# Convolution 2

def convolve2(ds, sigma):
    mpi_barrier()

    with MpiTimer('Three separable 1D convolutions') as T:
        pprint('Convolving axis Z', rank=0)

        ds1_ = ds.clone('gaussian_z')

        for z in range(mpi_rank(), ds.chunk_grid[0], mpi_size()):
            zS = z * ds.chunk_size[0]
            zE = min(zS + ds.chunk_size[0], ds.shape[0])
            out = gaussian_filter1d(ds[zS:zE], sigma=sigma, axis=0)
            ds1_[zS:zE] = out

        mpi_barrier()

        # Gaussian second axis

        pprint('Convolving axis Y', rank=0)

        ds2_ = ds.clone('gaussian_y')

        for y in range(mpi_rank(), ds.chunk_grid[1], mpi_size()):
            yS = y * ds.chunk_size[1]
            yE = min(yS + ds.chunk_size[1], ds.shape[1])
            out = gaussian_filter1d(ds1_[:, yS:yE], sigma=sigma, axis=1)
            ds2_[:, yS:yE] = out

        mpi_barrier()

        # Gaussian second axis

        pprint('Convolving axis X', rank=0)

        ds3_ = ds.clone('gaussian_x')

        for x in range(mpi_rank(), ds.chunk_grid[2], mpi_size()):
            xS = x * ds.chunk_size[2]
            xE = min(xS + ds.chunk_size[2], ds.shape[2])
            out = gaussian_filter1d(ds2_[..., xS:xE], sigma=sigma, axis=2)
            ds3_[..., xS:xE] = out

        mpi_barrier()

    if mpi_root():
        try:
            imsave(join(OUT_PATH, 'conv3x1d_{}-{}-{}.png'.format(
                   mpi_size(), ds.shape[0], ds.chunk_size[0])),
                   ds3_[ds.shape[0] // 2])
        except NameError:
            pass
    mpi_barrier()

    ds1_.delete()
    ds2_.delete()
    ds3_.delete()

    return T.time


###############################################################################
# Start tests!
if mpi_root():
    hout, dout = get_output_dataset()

for i, DS in enumerate(DATA_SIZE):

    f, data = create_random_dataset(DS)

    for j, CS in enumerate(CHUNK_SIZE):
        with dn.Connection(**CONNECTION_CONFIG) as connection:
            pprint('Loading Data -- shape: {} chunk_size: {}'
                   .format(DS, CS))
            with MpiTimer('Data loaded') as t:
                dataset = connection.create_dataset('data', data=data,
                                               chunk_size=(CS, CS, CS))

            for k in range(NTESTS):
                t1 = convolve1(dataset, SIGMA)
                t2 = convolve2(dataset, SIGMA)

                if mpi_root():
                    dout[i, j, 0, k] = t.time
                    dout[i, j, 1, k] = t1
                    dout[i, j, 2, k] = t2

            with MpiTimer('Data removed') as t:
                dataset.delete()

    f.close()
if mpi_root():
    hout.close()
