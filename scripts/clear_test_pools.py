

import sys
import argparse
import os.path as op

parent_folder = op.realpath(op.join(op.dirname(__file__), '..'))
sys.path.append(parent_folder)

import dosna as dn

if __name__ == '__main__':
    conffile = op.join(parent_folder, 'ceph.conf')

    parser = argparse.ArgumentParser(description='Remove temporary pools.')
    parser.add_argument('-r', '--remove-randoms', dest='remove_randoms', action='store_true')
    parser.set_defaults(remove_randoms=False)
    args = parser.parse_args()

    with dn.Cluster(conffile=conffile, timeout=5) as C:
        for pool in C.list_pools():
            if pool.name.startswith(C.__test_pool_prefix__):
                print("Removing", pool.name)
                pool.delete()
            elif pool.name.startswith(C.__random_pool_prefix__) and args.remove_randoms:
                print("Removing", pool.name)
                pool.delete()
            else:
                print(pool.name)
