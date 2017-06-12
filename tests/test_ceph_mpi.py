

import sys
import os.path as op

import rados
from dosna.mpi_utils import pprint, mpi_rank, mpi_barrier

parent_folder = op.realpath(op.join(op.dirname(__file__), '..'))

if __name__ == '__main__':
    if len(sys.argv) == 1:
        conffile = op.join(parent_folder, 'ceph.conf')
    else:
        conffile = sys.argv[1]

    cluster = rados.Rados(conffile=conffile)
    pprint("librados version: " + str(cluster.version()))

    cluster.connect(timeout=3)
    pprint("Cluster ID: " + cluster.get_fsid())

    if mpi_rank() == 0:
        cluster.create_pool('test_dosna')
    mpi_barrier()

    pprint(cluster.get_cluster_stats())
    pprint(cluster.pool_exists('test_dosna'))
