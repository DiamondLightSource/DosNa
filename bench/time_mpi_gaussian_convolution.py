

import time
import h5py as h5
import numpy as np
import os.path as op

from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter1d, gaussian_filter

import dosna.mpi as dn
dn.auto_init()

###############################################################################

DATA_SIZE = [128, 256, 512]
CHUNK_SIZE = [24, 36, 48, 60, 72, 96, 128]
NTESTS = 10

sigma = 3
trunc = 4

###############################################################################
# Input Data

in_path = op.realpath(op.join(op.dirname(__file__), '..', 'test_input.h5'))

def get_input_data(dsize):
    
    if dn.rank == 0:
        create = True
        if op.isfile(in_path):
            with h5.File(in_path, 'r') as f:
                create = f['data'].shape != (dsize, dsize, dsize)
            
        if create:
            dn.pprint('Writing random data', rank=0)
            with h5.File(in_path, 'w') as f:
                data = np.random.rand(dsize, dsize, dsize).astype(np.float32)
                f.create_dataset('data', data=data)
    
    dn.wait()
    
    h5in = h5.File(in_path, 'r')
    h5data = h5in['data']
    dn.pprint(h5data.shape)

    return h5in, h5data

###############################################################################
# Output Data

out_path = op.realpath(op.join(op.dirname(__file__), '..', 'test_result_batch.h5'))

def get_output_dataset(data_size, chunk_size):
    if dn.rank == 0:
        h5out = h5.File(out_path, 'a')
        
        mpath = 'mpi_%d' % dn.ncores
        dpath = 'data_%d' % data_size
        cpath = 'chunk_%d' % chunk_size
        
        if mpath not in h5out:
            h5out.create_group(mpath)
        if dpath not in h5out[mpath]:
            h5out[mpath].create_group(dpath)
        if cpath not in h5out[mpath][dpath]:
            h5result = h5out[mpath][dpath].create_dataset(cpath, shape=(3, NTESTS),
                                                          dtype=np.float32, fillvalue=0)
        else:
            h5result = h5out[mpath][dpath][cpath]
    else:
        h5out = h5result = None

    dn.wait()
    
    return h5out, h5result

###############################################################################
# Convolution 1

def convolve1(ds, sigma, chunk_size):
    dn.wait()
    t0 = time.time()
    
    dn.pprint('Convolving in 3D', rank=0)
    
    ds3d_ = ds.clone('gaussian_3d')
    
    for z in range(dn.rank, ds.chunks[0], dn.ncores):
        zS = z * ds.chunk_size[0]
        zE = min(zS + chunk_size, ds.shape[0])
        for y in range(ds.chunks[1]):
            yS = y * ds.chunk_size[1]
            yE = min(yS + chunk_size, ds.shape[1])
            for x in range(ds.chunks[2]):
                xS = x * ds.chunk_size[2]
                xE = min(xS + chunk_size, ds.shape[2])
                out = gaussian_filter(ds[zS:zE, yS:yE, xS:xE], sigma=sigma)
                ds3d_[zS:zE, yS:yE, xS:xE] = out
    
    dn.wait()
    t1 = time.time()
    dn.pprint('Separable 3D convolution performed in {0:.4f}'.format(t1 - t0), 
              rank=0)
    ds3d_.delete()
    return t1 - t0

###############################################################################
# Convolution 2

def convolve2(ds, sigma, chunk_size):
    # Gaussian First Axis
    dn.wait()
    t0 = time.time()
    
    dn.pprint('Convolving axis Z', rank=0)
    
    ds1_ = ds.clone('gaussian_z')
    
    for z in range(dn.rank, ds.chunks[0], dn.ncores):
        zS = z * ds.chunk_size[0]
        zE = min(zS + chunk_size, data.shape[0])
        out = gaussian_filter1d(data[zS:zE], sigma=sigma, axis=0)
        ds1_[zS:zE] = out
    
    dn.wait()
    
    # Gaussian second axis
    
    dn.pprint('Convolving axis Y', rank=0)
    
    ds2_ = ds.clone('gaussian_y')
    
    for y in range(dn.rank, ds.chunks[1], dn.ncores):
        yS = y * ds.chunk_size[1]
        yE = min(yS + chunk_size, data.shape[1])
        out = gaussian_filter1d(ds1_[:, yS:yE], sigma=sigma, axis=1)
        ds2_[:, yS:yE] = out
    
    dn.wait()
    
    # Gaussian second axis
    
    dn.pprint('Convolving axis X', rank=0)
    
    ds3_ = ds.clone('gaussian_x')
    
    for x in range(dn.rank, ds.chunks[2], dn.ncores):
        xS = x * ds.chunk_size[2]
        xE = min(xS + chunk_size, data.shape[2])
        out = gaussian_filter1d(ds2_[..., xS:xE], sigma=sigma, axis=2)
        ds3_[..., xS:xE] = out
    
    dn.wait()
    t1 = time.time()
    dn.pprint('Three separable 1D convolutions performed in {0:.4f}'
              .format(t1 - t0), rank=0)

    ds1_.delete()
    dn.wait()
    ds2_.delete()
    dn.wait()
    ds3_.delete()
    dn.wait()

    return t1 - t0

###############################################################################
# Start tests!

pool = dn.Pool()

for data_size in DATA_SIZE:
    
    f, data = get_input_data(data_size)
    
    for chunk_size in CHUNK_SIZE:
        dn.pprint('Loading Data -- shape: {} chunks: {}'
                  .format(data_size, chunk_size), rank=0)
        dn.wait()
        t0 = time.time()
        ds = pool.create_dataset('data', data=data, chunks=chunk_size)
        t1 = time.time()
        dn.pprint('Done in {}.'.format(t1 - t0), rank=0)
        
        g, out = get_output_dataset(data_size, chunk_size)
        if dn.rank == 0:
            out.attrs['data_size'] = data_size
            out.attrs['chunk_size'] = chunk_size
        
        t0 = t1 - t0
        
        for i in range(NTESTS):
            t1 = convolve1(ds, sigma, chunk_size)
            t2 = convolve2(ds, sigma, chunk_size)
            
            if dn.rank == 0:
                out[0, i] = t0
                out[1, i] = t1
                out[2, i] = t2
        
        ds.delete()
        if dn.rank == 0:
            g.close()
    
    f.close()
    
