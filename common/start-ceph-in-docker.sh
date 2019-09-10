#!/bin/bash
: ${BACKEND:=ceph}
if [[ "$BACKEND" != "ceph" ]]; then
    echo "Ceph is not required for this configuration"
    exit 0
fi
SUBNET="172.18.0.0/24"
IP="172.18.0.2"
TIMEOUT=30
docker pull ceph/daemon
docker network create --subnet $SUBNET testnet
echo "Starting Ceph"
container=$(docker run -d --ip $IP --net=testnet -e MON_IP=$IP -e MON_NAME=mon_node -e CEPH_PUBLIC_NETWORK=${SUBNET} ceph/daemon)
current_iteration=0
while true; do
    ceph_status=$(docker exec $container ceph status 2>&1)
    if [[ "$ceph_status" == *"HEALTH_OK"* ]]; then
        break
    fi
    if [[ "$ceph_status" == *"not running"* ]]; then
        echo "Error while starting ceph" 
        exit -1
    fi
    let current_iteration=current_iteration+1
    if [[ $current_iteration -ge $TIMEOUT ]]; then
        echo "Timeout while starting ceph"
        exit -1
    fi
    sleep 1
done
echo "Ceph started"
mkdir -p /etc/ceph
docker cp ${container}:/etc/ceph/ceph.conf .
docker cp ${container}:/etc/ceph/ceph.client.admin.keyring /etc/ceph
chmod +r /etc/ceph/ceph.client.admin.keyring
chmod +r ceph.conf
rados mkpool test-dosna
