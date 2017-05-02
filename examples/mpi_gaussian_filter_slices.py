

import time
import h5py as h5
import numpy as np
import os.path as op

from scipy.ndimage import gaussian_filter1d, gaussian_filter

import dosna.mpi as dn
dn.auto_init()

CSIZE = 64
SIGMA = 3
TRUNC = 4
RADIUS = int(SIGMA * TRUNC + 0.5)

###############################################################################
# Data

path = op.realpath(op.join(op.dirname(__file__), '..', 'test.h5'))
h5f = h5.File(path, 'r')
data = h5f['data']
dn.pprint(data.shape)

###############################################################################
# Load Data

pool = dn.Pool()

dn.pprint('Loading Data', rank=0)

ds = pool.create_dataset('data', data=data, chunks=CSIZE)

dn.pprint('Done.', rank=0)

###############################################################################
# Gaussian 3D

dn.wait()
t0 = time.time()

dn.pprint('Convolving in 3D', rank=0)

ds3d_ = ds.clone('gaussian_3d')

for z in range(dn.rank, ds.chunks[0], dn.ncores):
    zS = z * ds.chunk_size[0]
    zE = min(zS + CSIZE, ds.shape[0])
    for y in range(ds.chunks[1]):
        yS = y * ds.chunk_size[1]
        yE = min(yS + CSIZE, ds.shape[1])
        for x in range(ds.chunks[2]):
            xS = x * ds.chunk_size[2]
            xE = min(xS + CSIZE, ds.shape[2])
            out = gaussian_filter(ds[zS:zE, yS:yE, xS:xE], sigma=SIGMA)
            ds3d_[zS:zE, yS:yE, xS:xE] = out

dn.wait()
t1 = time.time()
dn.pprint('Separable 3D convolution performed in {0:.4f}'.format(t1 - t0), rank=0)

###############################################################################
# Separable Convolutions

# Gaussian First Axis

dn.wait()
t0 = time.time()

dn.pprint('Convolving axis Z', rank=0)

ds1_ = ds.clone('gaussian_z')

for z in range(dn.rank, ds.chunks[0], dn.ncores):
    zS = z * ds.chunk_size[0]
    zE = min(zS + CSIZE, data.shape[0])
    out = gaussian_filter1d(data[zS:zE], sigma=SIGMA, axis=0)
    ds1_[zS:zE] = out

dn.wait()

# Gaussian second axis

dn.pprint('Convolving axis Y', rank=0)

ds2_ = ds.clone('gaussian_y')

for y in range(dn.rank, ds.chunks[1], dn.ncores):
    yS = y * ds.chunk_size[1]
    yE = min(yS + CSIZE, data.shape[1])
    out = gaussian_filter1d(ds1_[:, yS:yE], sigma=SIGMA, axis=1)
    ds2_[:, yS:yE] = out

dn.wait()

# Gaussian second axis

dn.pprint('Convolving axis X', rank=0)

ds3_ = ds.clone('gaussian_x')

for x in range(dn.rank, ds.chunks[2], dn.ncores):
    xS = x * ds.chunk_size[2]
    xE = min(xS + CSIZE, data.shape[2])
    out = gaussian_filter1d(ds2_[..., xS:xE], sigma=SIGMA, axis=2)
    ds3_[..., xS:xE] = out

dn.wait()
t1 = time.time()
dn.pprint('Three separable 1D convolutions performed in {0:.4f}'
          .format(t1 - t0), rank=0)

###############################################################################
# Separable Convolutions in a Loopy version

dn.wait()
t0 = time.time()

ds_sep = ds.clone('gaussian_inplace')
prev = ds

for axis in range(ds.ndim):

    dn.pprint('Convolving axis {}'.format(['Z','Y','X'][axis]), rank=0)

    for i in range(dn.rank, ds.chunks[axis], dn.ncores):
        iS = i * ds.chunk_size[axis]
        iE = min(iS + CSIZE, data.shape[axis])
        slices = [slice(None)] * axis + [slice(iS, iE)]

        out = gaussian_filter1d(prev[slices], sigma=SIGMA, axis=axis)
        ds_sep[slices] = out

    prev = ds_sep
    dn.wait()

t1 = time.time()
dn.pprint('Three inplace separable 1D convolutions performed in {0:.4f}'
          .format(t1 - t0), rank=0)

###############################################################################
# Separable Convolutions in a Loopy version with padding

dn.wait()
t0 = time.time()

prev = ds

for axis in range(ds.ndim):
    dn.pprint('Convolving axis {}'.format(['Z','Y','X'][axis]), rank=0)

    ds_pad_n = ds.clone('gaussian_inplace_padding_{}'.format(axis))
    for i in range(dn.rank, ds.chunks[axis], dn.ncores):
        iS = i * ds.chunk_size[axis]
        iE = min(iS + CSIZE, data.shape[axis])
        pS = min(RADIUS, iS)
        pE = min(RADIUS, data.shape[axis] - iE)

        slices_in = [slice(None)] * axis + [slice(iS-pS, iE+pE)]
        slices_out_g = [slice(None)] * axis + [slice(iS, iE)]
        slices_out_l = [slice(None)] * axis + [slice(pS, CSIZE+pS)]

        out = gaussian_filter1d(prev[slices_in], sigma=SIGMA, axis=axis)
        ds_pad_n[slices_out_g] = out[slices_out_l]

    dn.wait()
    prev = ds_pad_n

t1 = time.time()
dn.pprint('Three separable 1D convolutions with padding performed in {0:.4f}'
          .format(t1 - t0), rank=0)

###############################################################################
# Separable Convolutions in a Loopy version with padding

dn.wait()
t0 = time.time()

ds_pad = ds.clone('gaussian_inplace_padding')
prev = ds

for axis in range(ds.ndim):
    dn.pprint('Convolving axis {}'.format(['Z','Y','X'][axis]), rank=0)

    for i in range(dn.rank, ds.chunks[axis], dn.ncores):

        iS = i * ds.chunk_size[axis]
        iE = min(iS + CSIZE, data.shape[axis])
        pS = min(RADIUS, iS)
        pE = min(RADIUS, data.shape[axis] - iE)

        slices_in = [slice(None)] * axis + [slice(iS-pS, iE+pE)]
        slices_out_g = [slice(None)] * axis + [slice(iS, iE)]
        slices_out_l = [slice(None)] * axis + [slice(pS, CSIZE+pS)]

        out = gaussian_filter1d(prev[slices_in], sigma=SIGMA, axis=axis)
        ds_pad[slices_out_g] = out[slices_out_l]

    prev = ds_pad
    dn.wait()

t1 = time.time()
dn.pprint('Three inplace separable 1D convolutions with padding performed in {0:.4f}'
          .format(t1 - t0), rank=0)

###############################################################################
# Full Convolution with padding - 1 Thread

dn.pprint('Convolving in 3D', rank=0)

dn.wait()
t0 = time.time()

ds_conv = ds.clone('gaussian_conv')

if dn.rank == 0:
    ds_conv[...] = gaussian_filter(ds[...], sigma=SIGMA)

dn.wait()
t1 = time.time()
dn.pprint('Full convolution in a single thread performed in {0:.4f}'
          .format(t1 - t0), rank=0)

###############################################################################
# Full Convolutions with padding - N Threads

dn.pprint('Convolving in 3D', rank=0)

dn.wait()
t0 = time.time()

ds_conv_n = ds.clone('gaussian_conv_n')

for z in range(dn.rank, ds.chunks[0], dn.ncores):
    zS = z * ds.chunk_size[0]
    zE = min(zS + CSIZE, data.shape[0])
    pS = min(RADIUS, zS)
    pE = min(RADIUS, data.shape[0] - zE)

    slices_in = slice(zS-pS, zE+pE)
    slices_out_g = slice(zS, zE)
    slices_out_l = slice(pS, CSIZE+pS)

    out = gaussian_filter(ds[slices_in], sigma=SIGMA)
    ds_conv_n[slices_out_g] = out[slices_out_l]

dn.wait()
t1 = time.time()
dn.pprint('Full convolution multi-thread performed in {0:.4f}'
          .format(t1 - t0), rank=0)


###############################################################################
# Verify results

if dn.rank == 0:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5*3, 4*3))
    plt.subplot(2,5,1); plt.imshow(ds1_[50], 'gray');
    plt.subplot(2,5,2); plt.imshow(ds2_[50], 'gray');
    plt.subplot(2,5,3); plt.imshow(ds3_[50], 'gray');
    plt.subplot(2,5,4); plt.imshow(ds_sep[50], 'gray');
    plt.subplot(2,5,5); plt.imshow(ds3d_[50], 'gray');
    plt.subplot(2,5,6); plt.imshow(ds_pad[50], 'gray');
    plt.subplot(2,5,7); plt.imshow(ds_pad_n[50], 'gray');
    plt.subplot(2,5,8); plt.imshow(ds_conv[50], 'gray');
    plt.subplot(2,5,9); plt.imshow(ds_conv_n[50], 'gray');
    plt.savefig('result.png', bbox_inches='tight')


assert np.allclose(ds3d_[...], ds3_[...])
assert np.allclose(ds3d_[...], ds_sep[...])
assert np.allclose(ds3_[...], ds_sep[...])
assert np.allclose(ds_pad_n[...], ds_conv[...])
assert np.allclose(ds_conv[...], ds_conv_n[...])
# They should not match due to overlapping patch rewriting while reading (inplace + padding)
assert not np.allclose(ds_pad[...], ds_conv[...])
print("Results do match!")

###############################################################################
# Finish

h5f.close()
pool.close()
pool.delete()
