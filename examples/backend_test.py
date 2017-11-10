

import logging as log
import sys

import numpy as np

import dosna as dn
from dosna.util.mpi import MpiTimer, mpi_barrier, mpi_comm, mpi_root, pprint

log.getLogger().setLevel(log.CRITICAL)

DS = 256
CS = 50

if __name__ == '__main__':

    dn.use(engine=sys.argv[1], backend=sys.argv[2])
    engine, backend = dn.status(show=True)

    params = dict(conffile=sys.argv[3]) if len(sys.argv) > 3 else {}

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    if mpi_root():
        rdata = np.random.rand(DS, DS, DS).astype(np.float32)
    else:
        rdata = None
    rdata = mpi_comm().bcast(rdata, root=0)

    with MpiTimer('DosNa (engine: %s, backend: %s)' %
                  (engine.name, backend.name)):

        pprint("Connecting", rank=0)
        with dn.Cluster('Cluster', **params) as C:
            pprint("Connected", C.connected)

            with C.create_pool('test_dosna') as P:
                pprint("Pool Created", rank=0)
                with MpiTimer('Dataset Created'):
                    data = P.create_dataset('test_data', data=rdata,
                                            chunks=(CS, CS, CS))
                if mpi_root():
                    pprint('Asserting the quality')
                    for i, j in [(0, 30), (10, 50), (5, 25), (37, 91)]:
                        np.testing.assert_allclose(data[i:j, i:j, i:j],
                                                   rdata[i:j, i:j, i:j])

                    print(data[25, 25, 25:35])
                    data[25:75, 25:75, 25:75] = np.random.rand(50, 50, 50)
                    print(data[25, 25, 25:35])

                    data[25:75, 25:75, 25:75] = rdata[25:75, 25:75, 25:75]
                    np.testing.assert_allclose(data[...], rdata[...])
                mpi_barrier()

            with C['test_dosna'] as P:
                P.del_dataset('test_data')
                pprint('Dataset Destroyed', rank=0)

            C.del_pool('test_dosna')
            pprint("Pool Destroyed", rank=0)
