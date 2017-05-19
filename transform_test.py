    

import h5py as h5
import numpy as np


with h5.File('test_result.h5', 'r') as f, h5.File('test_result_vol.h5', 'w') as g:

    core_keys = f.keys()
    data_keys = f['mpi_1'].keys()
    chunk_keys = f['mpi_1/data_128'].keys()
    
    ncores = len(core_keys)
    ndatas = len(data_keys)
    nchunk = len(chunk_keys)
    ntests = f['mpi_1/data_128/chunk_128'].size
    
    result = np.zeros((ncores, ndatas, nchunk, ntests), np.float32)
    for x1, f1 in enumerate(core_keys):
        for x2, f2 in enumerate(data_keys):
            for x3, f3 in enumerate(chunk_keys):
                if f3 in f[f1][f2]:
                    result[x1, x2, x3, :] = f[f1][f2][f3][:]

    g.create_dataset('data', data=result)
    g.create_dataset('mpi_jobs', data=np.asarray(core_keys).astype('|S9'))
    g.create_dataset('data_size', data=np.asarray(data_keys).astype('|S9'))
    g.create_dataset('chunk_size', data=np.asarray(chunk_keys).astype('|S9'))