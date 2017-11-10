

import logging as log

import numpy as np
from itertools import product

import dosna as dn
from dosna.util import Timer

log.getLogger().setLevel(log.INFO)

DS = 256
CS = 50

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    rdata = np.random.rand(DS, DS, DS).astype(np.float32)

    engines, backends = dn.engines.available, dn.backends.available

    for engine, backend in product(engines, backends):
        dn.use(engine=engine, backend=backend)
        engine, backend = dn.status(show=True)

        if not dn.compatible(engine, backend):
            log.warning('Skipping engine {} ({}) with {} backend'
                        .format(engine.name.capitalize(),
                                engine.params,
                                backend.name.capitalize()))
            continue

        with Timer('DosNa (engine: %s, backend: %s)' %
                   (engine.name, backend.name)):

            with dn.Cluster('/tmp/') as C:

                with C.create_pool('test_dosna') as P:
                    data = P.create_dataset(
                        'test_data', data=rdata, chunks=(CS, CS, CS))

                    for i, j in [(0, 30), (10, 50), (5, 25), (37, 91)]:
                        np.testing.assert_allclose(
                            data[i:j, i:j, i:j], rdata[i:j, i:j, i:j])

                    print(data[25, 25, 25:35])
                    data[25:75, 25:75, 25:75] = np.random.rand(50, 50, 50)
                    print(data[25, 25, 25:35])

                    data[25:75, 25:75, 25:75] = rdata[25:75, 25:75, 25:75]
                    np.testing.assert_allclose(data[...], rdata[...])

                with C['test_dosna'] as P:
                    P.del_dataset('test_data')

                C.del_pool('test_dosna')
