

import sys
import os.path as op

import dosna as dn
dn.use(engine='cpu', backend='ceph')

parent_folder = op.realpath(op.join(op.dirname(__file__), '..'))

if __name__ == '__main__':
    if len(sys.argv) == 2:
        conffile = op.join(parent_folder, 'ceph.conf')
    else:
        conffile = sys.argv[2]

    if len(sys.argv) <= 1:
        sys.exit(1)

    with dn.Cluster('Dummy', conffile=conffile, timeout=5) as C:
        for pool in sys.argv[1:]:
            print('Deleting pool', pool)
            C.del_pool(pool)
