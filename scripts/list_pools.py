

import sys
import argparse
import os.path as op

parent_folder = op.realpath(op.join(op.dirname(__file__), '..'))
sys.path.append(parent_folder)

import dosna as dn

if __name__ == '__main__':
    conffile = op.join(parent_folder, 'ceph.conf')

    with dn.Cluster(conffile=conffile, timeout=5) as C:
        for pool in C.list_pools():
            print(pool.name)