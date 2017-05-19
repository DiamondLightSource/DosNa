

import logging as log
log.getLogger().setLevel(log.DEBUG)

import numpy as np
import dosna.cpu as dn


with dn.Cluster('/tmp/') as C:
    
    with C.create_pool('test_dosna') as P:
        data = P.create_dataset('test_data', data=np.random.rand(100,100,100),
                                chunks=(10,10,10))
        
    
    #C.del_pool('test_dosna')