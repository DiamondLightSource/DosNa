

import logging as log
log.getLogger().setLevel(log.INFO)

import numpy as np
import dosna as dn
from dosna.utils import Timer

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
rdata = np.random.rand(100,100,100)

for backend in dn.backends.available:
    dn.use(backend=backend)
    engine, backend = dn.status(show=True)

    with Timer('DosNa (engine: %s, backend: %s)' % (engine.name, backend.name)):
        
        with dn.Cluster('/tmp/') as C:
        
            with C.create_pool('test_dosna') as P:
                data = P.create_dataset('test_data', data=rdata, chunks=(50,50,50))
        
                for i, j in [(0, 30), (10, 50), (5, 25), (37, 91)]:
                    np.testing.assert_allclose(data[i:j,i:j,i:j], rdata[i:j,i:j,i:j])
                
                print(data[0,0,17:25])
                data[0,0,17:25] = -5
                print(data[0,0,17:25])
                np.testing.assert_allclose(data[0,0,17:25], -5)
                
            with C['test_dosna'] as P:
                P.del_dataset('test_data')
        
            C.del_pool('test_dosna')