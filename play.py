import rados, sys

cluster = rados.Rados(conffile='ceph.conf')
print "\nlibrados version: " + str(cluster.version())
print "Will attempt to connect to: " + str(cluster.conf_get('mon initial members'))

cluster.connect()
print "\nCluster ID: " + cluster.get_fsid()

print "\n\nCluster Statistics"
print "=================="

cluster_stats = cluster.get_cluster_stats()

for key, value in cluster_stats.iteritems():
    print key, value


print "\nAvailable Pools"
print "----------------"

if not cluster.pool_exists('imanol-test'):
    cluster.create_pool('imanol-test')

pools = cluster.list_pools()

for pool in pools:
    print pool

print "\nOpening pool imanol-test"
ioctx = cluster.open_ioctx('imanol-test')

print "\nList files"
print "============"

object_iterator = ioctx.list_objects()

objects = list(object_iterator)

print "Total:", len(objects)

ioctx.close()

cluster.shutdown()

