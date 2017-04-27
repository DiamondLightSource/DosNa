

import time
import numpy as np
from mpi4py import MPI

import dosna as dn
dn.auto_init(njobs=1)


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print("Hello World! {} of {}".format(rank, size))

if rank == 0:
    print("({}) Creating random data".format(rank))
    random_data = np.random.rand(512, 512, 512)

    print("({}) Loading random data".format(rank))
    t0 = time.time()
    pool = dn.Pool()
    ds = pool.create_dataset('data', data=random_data, chunks=64, njobs=size)
    t1 = time.time()
    print("({0}) Done: {1:.4f}.".format(rank, t1 - t0))

    pool_name = pool.name
    ds_name = ds.name

    nchunks = ds.total_chunks
else:
    pool_name = None
    ds_name = None
    nchunks = None

pool_name = comm.bcast(pool_name, root=0)
ds_name = comm.bcast(ds_name, root=0)
nchunks = comm.bcast(nchunks, root=0)

pool = dn.Pool(pool_name)
ds = dn.Dataset(pool, ds_name)

print("({}) Pool: {} & Dataset: {}, {}, {}"
      .format(rank, pool.name, ds.name, ds.shape, ds.dtype))

# Read all the chunks
t0 = time.time()
total = 0
for idx in range(rank, nchunks, size):
    data = ds.get_chunk_data(idx)
    total += 1
t1 = time.time()
print("({0}) Time on reading {1} chunks: {2:.4f}".format(rank, total, t1 - t0))

# Over-write all the chunks
t0 = time.time()
total = 0
for idx in range(rank, nchunks, size):
    ds.set_chunk_data(idx, np.random.rand(*ds.chunk_size))
    total += 1
t1 = time.time()
print("({0}) Time on writing {1} chunks: {2:.4f}".format(rank, total, t1 - t0))

comm.Barrier()

if rank == 0:
    pool.delete()
