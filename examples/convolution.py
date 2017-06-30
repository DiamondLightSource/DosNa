

from __future__ import print_function


import sys
from os.path import join
import numpy as np
import h5py as h5

import tempfile

import dosna as dn
from dosna.mpi_utils import pprint, mpi_comm, mpi_rank, \
    mpi_size, mpi_root, mpi_barrier, MpiTimer

from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.misc import imsave

###############################################################################
# Test parameters

#DATA_SIZE = [128, 256, 512]
#CHUNK_SIZE = [24, 36, 48, 60, 72, 96, 128]
#NTESTS = 10

DATA_SIZE = [128]
CHUNK_SIZE = [60]
NTESTS = 10

SIGMA = 3
TRUNC = 4

if len(sys.argv) != 5:
    raise ValueError('Invalid number of arguments: '
                     '[backend, clustercfg, pool, output] '
                     'expected')

dn.use(backend=sys.argv[1], engine='mpi')

CLUSTERCFG = sys.argv[2]
POOL = sys.argv[3]
OUT_PATH = sys.argv[4]

engine, backend = dn.status()
pprint('Starting Test == Backend: {}, Engine: {}, Cluster: {}, Pool: {}, Out: {}'
       .format(backend.name, engine.name, CLUSTERCFG, POOL, OUT_PATH), rank=0)


###############################################################################
# Load data

def create_random_dataset(DS):
    pprint('Populating random data on disk', rank=0)

    if mpi_root():
        f = tempfile.NamedTemporaryFile()
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
    dname = '{}/{}/{}'.format(engine.name, backend.name, mpi_size())
    fname = join(OUT_PATH, 'result.h5')
    if mpi_root():
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
                g[dname].attrs['axis2_value'] = np.array(['3D', '1D'], dtype='S4')
                g[dname].attrs['axis3_value'] = list(range(NTESTS))

    mpi_barrier()

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

        for z in range(mpi_rank(), ds.chunks[0], mpi_size()):
            zS = z * ds.chunk_size[0]
            zE = min(zS + ds.chunk_size[0], ds.shape[0])
            for y in range(ds.chunks[1]):
                yS = y * ds.chunk_size[1]
                yE = min(yS + ds.chunk_size[1], ds.shape[1])
                for x in range(ds.chunks[2]):
                    xS = x * ds.chunk_size[2]
                    xE = min(xS + ds.chunk_size[2], ds.shape[2])
                    out = gaussian_filter(ds[zS:zE, yS:yE, xS:xE], sigma=sigma)
                    ds3d_[zS:zE, yS:yE, xS:xE] = out
        mpi_barrier()

    imsave(join(OUT_PATH, 'conv3d_{}-{}-{}.png'.format(
        mpi_size(), ds.shape[0], ds.chunk_size[0])), ds3d_[ds.shape[0] // 2])
    ds3d_.delete()

    return T.time


###############################################################################
# Convolution 2

def convolve2(ds, sigma):
    mpi_barrier()

    with MpiTimer('Three separable 1D convolutions') as T:
        pprint('Convolving axis Z', rank=0)

        ds1_ = ds.clone('gaussian_z')

        for z in range(mpi_rank(), ds.chunks[0], mpi_size()):
            zS = z * ds.chunk_size[0]
            zE = min(zS + ds.chunk_size[0], ds.shape[0])
            out = gaussian_filter1d(ds[zS:zE], sigma=sigma, axis=0)
            ds1_[zS:zE] = out

        mpi_barrier()

        # Gaussian second axis

        pprint('Convolving axis Y', rank=0)

        ds2_ = ds.clone('gaussian_y')

        for y in range(mpi_rank(), ds.chunks[1], mpi_size()):
            yS = y * ds.chunk_size[1]
            yE = min(yS + ds.chunk_size[1], ds.shape[1])
            out = gaussian_filter1d(ds1_[:, yS:yE], sigma=sigma, axis=1)
            ds2_[:, yS:yE] = out

        mpi_barrier()

        # Gaussian second axis

        pprint('Convolving axis X', rank=0)

        ds3_ = ds.clone('gaussian_x')

        for x in range(mpi_rank(), ds.chunks[2], mpi_size()):
            xS = x * ds.chunk_size[2]
            xE = min(xS + ds.chunk_size[2], ds.shape[2])
            out = gaussian_filter1d(ds2_[..., xS:xE], sigma=sigma, axis=2)
            ds3_[..., xS:xE] = out

        mpi_barrier()

    imsave(join(OUT_PATH, 'conv3x1d_{}-{}-{}.png'.format(
        mpi_size(), ds.shape[0], ds.chunk_size[0])), ds3_[ds.shape[0] // 2])

    ds1_.delete()
    ds2_.delete()
    ds3_.delete()

    return T.time


###############################################################################
# Start tests!

hout, dout = get_output_dataset()

for i, DS in enumerate(DATA_SIZE):

    f, data = create_random_dataset(DS)

    for j, CS in enumerate(CHUNK_SIZE):
        with dn.Cluster(CLUSTERCFG) as C:
            if backend.name in ['ram', 'hdf5'] and not C.has_pool(POOL):
                C.create_pool(POOL)
            with C[POOL] as P:
                pprint('Loading Data -- shape: {} chunks: {}'
                       .format(DS, CS))
                with MpiTimer('Data loaded') as t:
                    ds = P.create_dataset('data', data=data, chunks=(CS, CS, CS))

                for k in range(NTESTS):
                    t1 = convolve1(ds, SIGMA)
                    t2 = convolve2(ds, SIGMA)

                    if mpi_root():
                        dout[i, j, 0, k] = t.time
                        dout[i, j, 1, k] = t1
                        dout[i, j, 2, k] = t2

    f.close()
hout.close()
