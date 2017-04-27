

import time
import numpy as np
from mpi4py import MPI

#import dosna as dn
from time_utils import Timer
import dosna.mpi as dn
dn.auto_init(njobs=1)

BLOCK = True
CHECK = False and BLOCK
CHUNK_SIZE = 50

########################################
# MPI FOR LOADING DATA
########################################

def wait():
    if BLOCK:
        comm.Barrier()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = np.random.rand(100, 100, 100).astype(np.float32)
else:
    data = np.empty((100, 100, 100), np.float32)
data = comm.bcast(data, root=0)

data2 = np.arange(data.size).reshape(data.shape).astype(np.float32)
dchunk = np.random.randn(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE).astype(np.float32)

########################################
# DOSNA AUTO-MPI
########################################

comm.Barrier()  # Enforce True barriers for proper timming

if rank == 0:
    t0 = time.time()

pool = dn.Pool(test=True)

# Create dataset

wait()

with Timer('{}) Create dataset'.format(rank)):
    ds = pool.create_dataset('data', data=data, chunks=CHUNK_SIZE)

# Read slices
wait()

with Timer('{}) Reading slices'.format(rank)):
    result = ds[...]

assert not CHECK or np.allclose(ds[...], data)

wait()

# Write slices

with Timer('{}) Writing slices'.format(rank)):
    ds[...] = data2

assert not CHECK or np.allclose(ds[...], data2)

wait()

# Check chunk consistency

with Timer('{}) Replace block'.format(rank)):
    ds.set_chunk_data(0, dchunk)

wait()

isclose = np.allclose(ds.get_chunk_data(0), dchunk)
isclose = comm.gather(isclose, root=0)
if rank == 0 and CHECK:
    assert any(isclose)

wait()

pool.close()

comm.Barrier()  # Enforce True barriers for proper timming

if rank == 0:
    t1 = time.time()
    print('TOTAL TIME: {}'.format(t1 - t0))

pool.delete()
