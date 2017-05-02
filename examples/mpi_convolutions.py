

import h5py as h5
import numpy as np
import os.path as op

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from scipy.signal import fftconvolve
from skimage import transform, data, io

import dosna.mpi as dn
dn.auto_init()

###############################################################################

SIGMA = 30
TRUNC = 4
CSIZE = 1024

###############################################################################

def convolve(data_in, data_out, kernel, mode='reflect'):
    def conv(chunk, kernel, padding=None):
        return fftconvolve(chunk, kernel, mode='valid')
    padding = [s//2 for s in kernel.shape]
    ds_out = data_in.map_padded(data_out, conv, padding, kernel, mode=mode)
    return ds_out

###############################################################################

pool = dn.Pool()

if dn.rank == 0:
    img = data.camera()
    img = transform.rescale(img, 8, mode='reflect')
    shape = img.shape
    dtype = img.dtype
else:
    img = None
    shape = None
    dtype = None

###############################################################################

ds = pool.create_dataset('data', dtype=dtype, shape=shape, chunks=CSIZE)

dn.pprint('Loading Dataset of shape', ds.shape, 'dtype', ds.dtype)

ds.load(img, use_mpi=False)

###############################################################################
# Generate 3D Gaussian Kernel

r = int(SIGMA * TRUNC + 0.5)
x = np.arange(-r, r+1)
g1d = np.exp(-x**2 / (2 * SIGMA**2)) /  np.sqrt(2.0 * np.pi)
g2d = g1d[:, None] * g1d[None, :]
g2d /= np.abs(g2d).sum() # Normalize

r = convolve(ds, 'ds_out', g2d)

if dn.rank == 0:
    io.imsave('result2.png', r[...])

