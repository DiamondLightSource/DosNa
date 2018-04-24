# Distributed Object-Store Numpy Array (DosNa)

[![Build Status](https://travis-ci.org/DiamondLightSource/DosNa.svg?branch=master)](https://travis-ci.org/DiamondLightSource/DosNa)

DosNa is intended to be a python wrapper to distribute N-dimensional arrays
over an Object Store server. The main goal of DosNa is to provide easy and
friendly seamless interface to store and manage N-Dimensional datasets over
a remote Cloud.

It is designed to be modular by defining a `Connection -> Dataset -> DataChunk`
architecture, and supports multiple **Backends** and **Engines** that
extend the base abstract model to add different functionality. Each Backend
represents a different Object Store type and wraps DosNa to connect and
interact with such Object Store. Engines, on the other hand, add different
local (or remote) multi-threading and multi-process support. Engines act
as clients to the selected Backend and parallelize (or enhance) some of
the functions to access the remote data.

For example, the Ceph backend works by directly wrapping
[`Cluster`](http://docs.ceph.com/docs/master/rados/api/python/#cluster-handle-api)
and [`Pool/IOCtx`](http://docs.ceph.com/docs/master/rados/api/python/#input-output-context-api)
API's from Librados and extending two new classes, `Dataset` and `DataChunk`
that wrap and extend the behavior of a Librados
[`Object`](http://docs.ceph.com/docs/master/rados/api/python/#object-interface)
(that are only accessible through looping the `Pool`).

![](/DOSNA.png)

DosNa is based on the [Librados architecture](http://docs.ceph.com/docs/master/rados/api/python/)
library and tries to mimic its model of an Object Store, thus `dosna.Cluster`
and `dosna.Pool` are connection objects to the remote Object Store service
and the Pools (or streams) within it.

A `dosna.Dataset` will *automagically* distribute a N-dimensional dataset
across the cluster by partitioning the whole data in smaller chunks and
store them in a completely distributed fashion as `dosna.DataChunks`.
After a `chunk_size` is specified, data loaded to a `dosna.Dataset` will
be split in to chunks of size `chunk_size` and will create different remote
objects as `dosna.DataChunk`, each of them corresponding to a different
chunk from the original data. These `dosna.DataChunk` objects are
distributed along the Object Store, making `dosna.Dataset` a wrapper
(acting as a lookup table) to distribute N-dimensional datasets into
many N-dimensional smaller chunks.

An existing `dosna.Dataset` can be used as an `h5py.Dataset` object or a
Numpy Array. This is, a dataset object supports standard slicing `ds[:, :, :]`
(getter) and `ds[:, :, :] = 5` (setter) and the `dosna.Dataset` object will
take care behind the scenes to access all the `dosna.DataChunk` needed to
reconstruct the desired slices and retrieve or update them accordingly.

## Installation

There are no requirements other than NumPy to install and use DosNa with
the default Backend and Engine. However, specific backend and engines
require different dependencies listed as follows:

Requirements:

 - Library: [numpy](http://www.numpy.org/)
 - Backends:
     + ram: none
     + hdf5: [h5py](http://docs.h5py.org/en/latest/quick.html)
     + ceph: [librados](http://docs.ceph.com/docs/master/rados/api/librados-intro/#getting-librados-for-python)
 - Engines:
     + cpu: none
     + jl: [joblib](https://pythonhosted.org/joblib/)
     + mpi: [mpi4py](http://mpi4py.scipy.org/)
     + sage: [pyclovis](/dosna/support/pyclovis)
 - Examples:
     + `convolutions.py`: [scipy](https://www.scipy.org/)

Clone repository:

```bash
git clone https://github.com/DiamondLightSource/DosNa.git
```

Install it:

```bash
cd DosNa && python setup.py install
```

Configure connection to a Ceph cluster (for using with Ceph backend):

 - Create a `ceph.conf` file with the IP or HOST addresses of the CEPH
 entry and cluster nodes (see `ceph.sample.conf` as an example)

## Basic Usage

Create a Pool object and a dataset. In the example bellow a dataset of
size `(100, 100, 100)` is created and

```python
import dosna as dn
import numpy as np

data = np.random.randn(100, 100, 100)

con = dn.Connection('dosna-tutorial')
con.connect()
ds = con.create_dataset('data', data=data, chunk_size=(32,32,32))

print(ds[0, 0, :10])
# [ 0.38475526 -1.02690848  0.88673305 -0.12398208  1.49337365 -1.91976844
#   1.76940625  0.58314611  0.41391879 -0.34711908]
assert np.allclose(ds[...], data) # True

con.disconnect()
```

The same example can also be run with context managers, for a more safe
access to the clusters and pools without forgetting to `close` corresponding
objects:

```python
data = np.random.randn(100, 100, 100)

with dn.connection('dosna_tutorial') as con:
    ds = con['data'] # Similarly, retrieve an existing dataset
    print(ds[0, 0, :10])
    # [ 0.38475526 -1.02690848  0.88673305 -0.12398208  1.49337365 -1.91976844
    #   1.76940625  0.58314611  0.41391879 -0.34711908]
    ds[36:80, 89:, :56] = data[36:80, 89:, :56]
    assert np.allclose(ds[36:80, 89:, :56], data[36:80, 89:, :56])
```

Note that the example above assumes that the pool `dosna_tutorial` already
exists. Pool objects are not expected to be created carelessly, instead,
a single (or few) pools are expected to exist within an Object Store, and
although DosNa could create/remove them, its intended usage is to make
use of existing pools.

## Selecting Backend and Engines

Selecting backend is very trivial and can be done at any point. However,
the backend wont change for already created `Cluster` instances. New
`Cluster` instances have to be created in order to see the change:

```python
dn.use(backend='hdf5') # One of ['ram', 'hdf5', 'ceph', 'sage']
```

Similarly, engines can be selected as:

```python
dn.use(engine='mpi') # One of ['cpu', 'joblib', 'mpi']
```

However, a Backend and Engine is not expected to change dynamically during
the execution of an script, and thus, although DosNa can handle does changes,
its expected usage is to define the Engine and Backend manually after the
import at the top of the script:

```python
import dosna as dn
dn.use(backend='hdf5', engine='mpi')

# Rest of the script
```

## Backends

There are currently 2 test backends (`ram` and `hdf5`) and 2 backend
wrapping a real object store (`ceph` and `sage`).

### Ram

The `ram` (default) backend simulates an on-memory Object Store by defining
the Connection as a standard python dictionary. Similarly, a Dataset use
a dictionary working as lookup table for all the existing chunks, with
additional attributes such as the dataset shape, the dataset data type
and the size of the chunks. DataChunks are then on-memory numpy arrays.

This backend is the quickest one and very useful to test different
functionality of DosNa. It is important to note that when used with MPI
Engine it will have an unpredictable weird behavior, as each process will
have its own copy of the data on memory. It will work with Joblib Engine
though, as long as the `multithreading` option is selected.

Connection parameters: \
`Connection(name, directory='.')`

- name (string): The name that identifies the connection.

### Hdf5

Simulates an on-disk Object Store by defining the Connection and Dataset
as an on-disk folder tree. DataChunks are then automatically created HDF5
datasets inside the *Dataset* folder, containing a `/data` dataset inside
them with the actual data.

This backend is useful for testing multi-processing Engines (Joblib with
`multiprocessing` and MPI) as the on-disk storage makes easy to distribute
`h5py.Dataset` instances over the processes without corrupting the underlying
data.

Although it is a *test* backend, it is probably very efficient and could
replace standard `h5py` in extremely parallel scenarios, as DataChunks
are separate HDF5 files and thus, it would allow for complete parallelization
(both read and write) as long as no 2 processes access the same DataChunk
simultaneously.

Connection parameters: \
`Connection(name, directory='.')`

- name (string): The directory name used to hold the HDF5 datasets,
the directory should exist.
- directory (string): The path where the name will be searched.
The current directory is used by default.

### Ceph

Ceph backend connects to a Ceph Cluster. Connection object act as a
connection to the cluster and a pool within it. Creation or deletion of
pools happens directly in the Object Store and similarly, a pool can
create, retrieve or remove Datasets, represented as Ceph Objects with
metadata and lookup information for the DataChunks that contain the
appropiate data. DataChunks are another type of Ceph Objects that
contain the numerical data in binarized numpy arrays format.

Connection parameters: \
`Connection(name, conffile='ceph.conf', timeout=5, client_id=None)`

- name (string): The ceph pool name, used for storing the dataset objects,
the pool should exist.
- conffile (string): The ceph configuration file path. Defaults to 'ceph.conf'
in current directory.
- timeout (int): The connection timeout. (Default: 5 seconds)
- client_id: The ceph client id, e.g: client.user. (Default: None)

## Engines

There are 3 different backends: `cpu`, `joblib` and `mpi`.


### Cpu

Although the name might be a bit misleading, it refers to a single process
single thread instance. This is, a sequential program with no parallelization
at all. It is the default engine as it is the easiest to test and maintain
and will act as a simple wrapper along the selected Backend.

### Jl (Joblib)

Will act as a local multi-threading or multi-processing backend. It
accepts `njobs` and `jlbackend` parameters, the first one indicating the
number of threads or processes to use while the second one indicating
whether to use multi-threading or multi-processing (values for `jlbackend`
are `threading` or `multiprocessing`).

This engine will act as the CPU backend for most of the functionallity,
but incorporates additional parallelization
when 1) slicing and 2) mapping/applying functions to all the chunks.

This is, when slicing a dataset `ds[10:50, 10:50, 10:50]`, the Dataset
instance will first calculate all the DataChunks involved in such slicing,
and then using joblib each thread/process will gather/update the data
from different DataChunks (acting as a map-reduce operator).

Similarly, a Dataset contains `ds.map` and `ds.apply` functions that map
(return a modified copy) or apply (in-place) a function to every chunk in
the dataset. This function will be parallelized by each thread/process
taking care of applying the function to a different chunk.

Last, a whole numpy or h5py array can be loaded into an object store, by
populating the corresponding DataChunks by doing `ds.load(data)`, this
function will also be parallelized by spawning threads or processes that
update different chunks.

### MPI

The MPI backend is designed to work fully in parallel by using multiple
processes. Different from Joblib, processes are not spawned when
functions are called, instead, multiple processes exist from the start
of the script and each of them contains a copy of the `Cluster` object.
From there, MPI backend adds wrappers to most of the creation/deletion
functions so that only the lead process (or root process, commonly the
one with rank 0) creates or deletes the resource while the others wait
for it. The engine also adds appropriate barriers when needed so that
every process contains the same state of the remote Object Store. Last,
the MPI Engine does not add parallelization over the slicing operation,
as it is assumed that the script will be programmed so that different
processes access different slices of the dataset. To that end,
`Connection` and `Dataset` objects contain `mpi_size`, `mpi_rank` and
`mpi_is_root` attributes when used with the MPI engine.

Similar to joblib, it also extends `dataset.load`, `dataset.map` and
`dataset.apply` so that automatically each of the process takes care
of different chunk.

## Creating a new Backend

Backends extend the `BaseCluster`, `BaseDataset` and `BaseDataChunk`
templates present at `dosna.base` class to add functionality regarding
the connection and creation of the respective objects in the new Object
Store.

The methods that have to be rewritten are:

- Connection
    + `connect` and `disconnect`
    + `create_dataset`, `get_dataset`, `has_dataset` and `del_dataset`
- Dataset
    + `create_chunk`, `get_chunk`, `has_chunk` and `del_chunk`.
- DataChunk
    + `get_data`, `set_data`

For quick examples of how to create a new backend have a look at
`dosna.backends.ram` or `dosna.backends.hdf5`.

## Creating a new Engine

Engines are sightly more complicated, as they don't directly extend the
`dosna.base` templates, but extend `dosna.base.Wrapper` class, which
creates wrapper objects around the currently selected Backend.

The Engine objects, as they act as wrappers, have to override all the
create/delete/get operations to return also the corresponding wrapped
object instead of the native Backend object. This is, `engine.Cluster`
wraps `backend.Cluster`, but has to redefine `create_pool` so that
`engine.Pool` is returned instead of `backend.Pool` when `create_pool`
is called (see `dosna.backends.cpu` for a quick example).

The list of methods modified to **properly wrap** a Backend are:

- Connection
    + `create_dataset`, `get_dataset`, `has_dataset` and `del_dataset`
- Dataset
    + `create_chunk`, `get_chunk`, `has_chunk` and `del_chunk`.
    + `clone` to clone a dataset (only its structure/chunk layout or
    also the underlying chunks)

The list of methods that can be extended to **add functionality** are:

- Dataset:
    + `get_data`, `set_data` to override slicing operations
    + `load` to load and distribute a whole dataset along DataChunks
    + `map` to map a function to all the chunks independently and return
    a modified copy of the dataset.
    + `apply` to apply a function to all the chunks in-place

## Running tests

### Unit Tests

For checking the general library functionality run tests as follows:

```bash
sh run_tests.sh
```

For checking specific backends or engines run add appropriate parameters:

```bash
sh run_tests.sh --backend <backend> --engine <engine>
```

Other parameters are available to setup backends, engines and pools
appropriately (check `sh run_tests.sh --help` for more info).

It is important to note that:

- `ram` backend doesn't work properly with Joblib and MPI engines.
- `hdf5` backend requires to provide an existing folder as `--connection`
parameter.
- `ceph` backend requires a pool name passed as `--connection` and a
configuration file passed as `--connection-options conffile=...`.

And last, to run tests with `mpi` the script has to be called as:

```bash
mpiexec -n <nproc> sh run_tests.sh
```

where `<nproc>` is an integer indicating the number of MPI cores to spawn.

### Quick backend testing

There are few examples for testing individual or multiple backends and
engines in the `examples` folder. Can be run as an standard python
program, or through `sh run_mpi.sh <nproc> <script>`, where script is
the path to the script to be run, if the MPI engine is going to be used.

### Convolution test

A test of computing 3D Gaussian Convolutions over an 3D dataset is
available at `examples/convolutions.py`. This script will create
datasets of different sizes with different chunk sizes and perform a 3D
convolution over the chunks, as well as performing three separable 1D
convolutions (with the same results). Timings for all convolutions for
different data size and data chunks will be stored on an output HDF5 file.

The script can be used to analyze the impact of chunk sizes and dataset
sizes on different backends and engines.

To use the script:

```bash
python examples/convolutions.py ... --out=<outdir>
```

To run it with the MPI engine:

```bash
sh run_mpi.sh <nproc> examples/convolutions.py ... --out=<outdir>
```

For further parameters specifying the engine, backend or other
parameters as data sizes or chunk sizes to be tested call it with `--help`.

The resulting HDF5 file with timings and couple of images for visual
verification will be stored in `<outdir>`.
