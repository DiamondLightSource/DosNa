# Distributed Object-Store Numpy Array (DosNa)

DosNa is intended to be a python wrapper around [Librados (Python)](http://docs.ceph.com/docs/master/rados/api/python/) library from [Ceph](http://ceph.com/). The main goal of DosNa is to provide easy and friendly interface to store and manage N-Dimensional datasets over a Ceph Cloud.

It works by directly wrapping [`Cluster`](http://docs.ceph.com/docs/master/rados/api/python/#cluster-handle-api) and [`Pool/IOCtx`](http://docs.ceph.com/docs/master/rados/api/python/#input-output-context-api) API's from Librados and extending two new classes, `Dataset` and `DataChunk` that wrap and extend the behavour of a Librados [`Object`](http://docs.ceph.com/docs/master/rados/api/python/#object-interface) (that are only accessible through looping the `Pool`).

![](/DOSNA.png)

DosNa is also inspired by the [h5py](http://www.h5py.org/) library and tries to mimic its behaviour, thus `dosna.Pool` and `dosna.Dataset` objects are equivalent objects to `h5py.File` and `h5py.Dataset` respectively (in fact, there is a `dosna.File = dosna.Pool` symlink to fully mimic h5py).

A `dosna.Dataset` will automagically distribute a N-dimensional dataset across the cluster by partitioning the whole data in smaller chunks and store them in a completely distributed fashion as `dosna.DataChunks`. After a `chunk_size` is specified, data loaded to a `dosna.Dataset` will be split in to chunks of size `chunk_size` and will create different Ceph objects as `dosna.DataChunk`, each of them corresponding to a different chunk from the original data. These `dosna.DataChunk` objects are distributed along the Ceph cluster, making `dosna.Dataset` a wrapper to distribute N-dimensional datasets into many N-dimensional smaller chunks.

An existing `dosna.Dataset` can be used as an `h5py.Dataset` object or a Numpy Array. This is, a dataset object supports standard slicing `ds[:, :, :]` and `ds[:, :, :] = 5` and the `dosna.Dataset` object will take care behind the scenes to access all the `dosna.DataChunk` needed to reconstruct the desired slices.

## Installation

Requirements: 

 - Library: Librados, numpy, joblib
 - Examples: Scipy, Matplotlib, Scikit-Image

Clone repository:

    git clone https://github.com/DiamondLightSource/DosNa.git

Install it:

    cd DosNa && python setup.py install

Configure cluster:

 - Create a `ceph.conf` file with the IP or HOST addresses of the CEPH entry and cluster nodes (see `ceph.sample.conf` as an example)
    
## Basica Usage

Create a Pool object and a dataset:

```python
import dosna as dn
import numpy as np

data = np.random.randn(100, 100, 100)

C = dn.Cluster().connect()
pool = C.create_pool('tutorial')
ds = pool.create_dataset('data', data=data, chunks=32) # chunking of int maps to dimensions -> (32, 32, 32)

print(ds[0, 0, :10])
# [ 0.38475526 -1.02690848  0.88673305 -0.12398208  1.49337365 -1.91976844
#   1.76940625  0.58314611  0.41391879 -0.34711908]
assert np.allclose(ds[...], data) # True

pool.close()
C.disconnect()
```
    
Same example can also be run with context managers, for a more safe access to the clusters and pools without forgeting to `close` corresponding objects:
 
```python
data = np.random.randn(100, 100, 100)

with dn.Cluster() as C:
    with C['tutorial'] as pool: # Note that __getitem__ retrieves an existing pool
        ds = pool['data'] # Similarly, retrieve an existing dataset
        print(ds[0, 0, :10])
        # [ 0.38475526 -1.02690848  0.88673305 -0.12398208  1.49337365 -1.91976844
        #   1.76940625  0.58314611  0.41391879 -0.34711908]
        ds[36:80, 89:, :56] = data[36:80, 89:, :56]
        assert np.allclose(ds[36:80, 89:, :56], data[36:80, 89:, :56])
```

## Using default Cluster

For programs working with a single `dosna.Cluster` object (as will be usual) the cluster instance can be completely abstracted by using `dosna.autoinit`. It will make every `dosna.Pool` instance connect to the singleton cluster instance.

```python
dn.auto_init()

with Pool('tutorial') as p:
    ds = p['data'] 
    [...]
```

## h5py compatibility

DosNa also shadows `dosna.Pool` as `dosna.File` for compatibilities with h5py, allowing the following piece of code to be run with either of libraries:

```python
import numpy as np

if True: # Change to False to run the same code with DosNa instead of h5py
    import h5py as h5
else:
    import dosna as h5
    h5.auto_init(njobs=1) # Change to number of jobs

with h5.File('/tmp/data.h5', 'w') as f: # since dosna.File subclasses dosna.Pull this will work in dosna
    f.create_dataset('data', data=np.random.rand(100, 100, 10), chunks=(32, 32, 32), fillvalue=-1)

f = h5.File('/tmp/data.h5', 'r')
data = f['data']
print(data[0, 0, :10])
f.close()
```

Note that the above script will run without errors using either h5py or dosna backend. See `examples/basic_h5py_compat.py` for more details.

## Multi threading

All `dosna.Cluster`, `dosna.Pool` and `dosna.Dataset` accept a `njobs` parameter (`1` or `None` by default) that is propagated through the childrens (from Cluster to Pool and Pool to Dataset) if not specified (`None`) in the children. It will parallelize the chunk gathering and setting over multiple threads using [joblib](https://pythonhosted.org/joblib/).
