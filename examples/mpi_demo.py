

from __future__ import print_function

import logging as log
import tempfile

import h5py as h5
import numpy as np

import dosna as dn
from dosna.util.mpi import (MpiTimer, mpi_comm, mpi_rank, mpi_root, mpi_size,
                            pprint)

log.getLogger().setLevel(log.INFO)

dn.use(engine='mpi', backend='hdf5')

###############################################################################

DS = 256
CS = 50

engine, backend = dn.status()

pprint('Loading data', rank=0)

if mpi_root():
    f = tempfile.NamedTemporaryFile()
    data = np.random.rand(DS, DS, DS).astype(np.float32)
    with h5.File(f.name, 'w') as g:
        g.create_dataset('data', data=data)
    fname = f.name
else:
    fname = None
fname = mpi_comm().bcast(fname, root=0)

h5in = h5.File(fname, 'r')
h5data = h5in['data']

###############################################################################

with MpiTimer('DosNa %s (%s)' % (engine.name, backend.name)):

    with dn.Cluster('/tmp/') as C:

        with C.create_pool('test_dosna') as P:
            data = P.create_dataset('test_data', data=h5data,
                                    chunks=(CS, CS, CS))

            for start in range(mpi_rank() * CS, DS, mpi_size() * CS):
                stop = start + CS
                i = start
                j = min(stop, DS)
                np.testing.assert_allclose(data[i:j], h5data[i:j])

            for start in range(mpi_rank() * CS, DS, mpi_size() * CS):
                stop = start + CS
                i = start
                j = min(stop, DS)
                np.testing.assert_allclose(data[:, i:j], h5data[:, i:j])

            for start in range(mpi_rank() * CS, DS, mpi_size() * CS):
                stop = start + CS
                i = start
                j = min(stop, DS)
                np.testing.assert_allclose(data[..., i:j], h5data[..., i:j])

        C.del_pool('test_dosna')

###############################################################################

if mpi_root():
    f.close()
