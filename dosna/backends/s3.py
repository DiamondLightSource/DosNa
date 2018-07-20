#!/usr/bin/env python
"""Backend s3 uses a S3 interface to store the dataset and chunks data"""

import logging

import numpy as np

import boto3
from botocore.exceptions import ClientError
from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset, ConnectionError,
                                 DatasetNotFoundError)
from dosna.util import dtype2str, shape2str, str2shape
from dosna.util.data import slices2shape

_DATASET_ROOT = 'dataset_root'
_SIGNATURE = "DosNa Dataset"

_SHAPE = 'shape'
_DTYPE = 'dtype'
_FILLVALUE = 'fillvalue'
_CHUNK_GRID = 'chunk-grid'
_CHUNK_SIZE = 'chunk-size'

log = logging.getLogger(__name__)

# Sanitise bucket name to conform to AWS conventions


def bucketName(name):
    return name.replace('_', '-').lower()


class S3Connection(BackendConnection):
    """
    A S3 Connection that wraps boto3 S3 client
    """

    def __init__(self, name, endpoint_url=None, verify=True,
                 profile_name='default',
                 *args, **kwargs):
        super(S3Connection, self).__init__(name, *args, **kwargs)

        self._endpoint_url = endpoint_url
        self._verify = verify
        self._client = None
        self._profile_name = profile_name

        super(S3Connection, self).__init__(name, *args, **kwargs)

    def connect(self):

        if self.connected:
            raise ConnectionError(
                'Connection {} is already open'.format(self.name))
        session = boto3.session.Session(profile_name=self._profile_name)

        # Use access key and secret_key in call to client?
        self._client = session.client(
            service_name='s3',
            endpoint_url=self._endpoint_url,
            verify=self._verify
        )

        # Check bucket exists and is writable

        super(S3Connection, self).connect()

    def disconnect(self):

        if self.connected:
            super(S3Connection, self).disconnect()

    @property
    def client(self):
        return self._client

    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunk_size=None):
        if not ((shape is not None and dtype is not None) or data is not None):
            raise Exception('Provide `shape` and `dtype` or `data`')
        if self.has_dataset(name):
            raise Exception('Dataset `%s` already exists' % name)

        if data is not None:
            shape = data.shape
            dtype = data.dtype

        if chunk_size is None:
            chunk_size = shape

        chunk_grid = (np.ceil(np.asarray(shape, float) / chunk_size))\
            .astype(int)

        name = bucketName(name)

        log.debug('creating dataset %s with shape:%s chunk_size:%s '
                  'chunk_grid:%s', name, shape, chunk_size, chunk_grid)

        try:
            self._client.create_bucket(Bucket=name, ACL='private')
        except ClientError as e:
            code = e.response['Error']['Code']
            if code is not None:
                log.error('connect: create_bucket returns %s', code)
                return None

        metadata = {
            _SHAPE: shape2str(shape),
            _DTYPE: dtype2str(dtype),
            _FILLVALUE: repr(fillvalue),
            _CHUNK_GRID: shape2str(chunk_grid),
            _CHUNK_SIZE: shape2str(chunk_size)
        }

        self._client.put_object(
            Bucket=name, Key=_DATASET_ROOT,
            Body=bytes(_SIGNATURE), Metadata=metadata
        )

        dataset = S3Dataset(
            self, name, shape, dtype,
            fillvalue, chunk_grid, chunk_size
        )

        return dataset

    def get_dataset(self, name):

        if not self.has_dataset(name):
            raise DatasetNotFoundError('Dataset `%s` does not exist' % name)

        metadata = self._dataset_root['Metadata']
        if metadata is None:
            raise DatasetNotFoundError(
                'Dataset `%s` does not have required DosNa metadata' % name
            )

        shape = str2shape(metadata[_SHAPE])
        dtype = metadata[_DTYPE]
        fillvalue = int(metadata[_FILLVALUE])
        chunk_grid = str2shape(metadata[_CHUNK_GRID])
        chunk_size = str2shape(metadata[_CHUNK_SIZE])
        dataset = S3Dataset(
            self, name, shape, dtype, fillvalue,
            chunk_grid, chunk_size
        )

        return dataset

    def get_dataset_root(self, name):

        name = bucketName(name)

        dataset_root = None
        try:
            dataset_root = self._client.get_object(
                Bucket=name, Key=_DATASET_ROOT
            )

            content = dataset_root['Body'].read()
            if not content == _SIGNATURE:
                dataset_root = None

        except Exception:
            pass  # Don't need to report errors here

        return dataset_root

    def has_dataset(self, name):

        self._dataset_root = self.get_dataset_root(name)
        if self._dataset_root is None:
            log.info("has_dataset: dataset %s does not exist", name)

        return self._dataset_root is not None

    def del_dataset(self, name):

        if self.has_dataset(name):

            name = bucketName(name)
            try:
                self._client.delete_object(Bucket=name, Key=_DATASET_ROOT)
                self._client.delete_bucket(Bucket=name)
            except ClientError as e:
                log.error('del_dataset: cannot delete %s: %s',
                          name, e.response['Error'])
        else:
            raise DatasetNotFoundError(
                'Dataset `{}` does not exist'.format(name))


class S3Dataset(BackendDataset):
    """
    S3Dataset
    """

    @property
    def client(self):
        return self.connection.client

    def _idx2name(self, idx):
        return '.'.join(map(str, idx))

    def create_chunk(self, idx, data=None, slices=None):
        if self.has_chunk(idx):
            raise Exception('DataChunk `{}{}` already exists'.
                            format(self.name, idx))
        name = self._idx2name(idx)
#        print "Name = %s" % (name)
        dtype = self.dtype
        shape = self.chunk_size
        fillvalue = self.fillvalue
        datachunk = S3DataChunk(self, idx, name, shape, dtype, fillvalue)
        if data is None:
            data = np.full(shape, fillvalue, dtype)
        datachunk.set_data(data, slices, fill_others=True)
        return datachunk

    def get_chunk(self, idx):
        if self.has_chunk(idx):
            name = self._idx2name(idx)
            dtype = self.dtype
            shape = self.chunk_size
            fillvalue = self.fillvalue
            return S3DataChunk(self, idx, name, shape, dtype, fillvalue)
        return self.create_chunk(idx)

    def has_chunk(self, idx):

        has_chunk = False
        name = self._idx2name(idx)
        try:
            self.client.head_object(Bucket=bucketName(self._name), Key=name)
            has_chunk = True
        except ClientError as e:
            log.debug("ClientError: %s", e.response['Error']['Code'])

        return has_chunk

    def del_chunk(self, idx):
        if self.has_chunk(idx):
            self.client.delete_object(
                Bucket=bucketName(self._name),
                Key=self._idx2name(idx)
            )


class S3DataChunk(BackendDataChunk):

    @property
    def client(self):
        return self.dataset.client

    def get_data(self, slices=None):
        if slices is None:
            slices = slice(None)
        data = np.fromstring(self.read(), dtype=self.dtype, count=self.size)
        data.shape = self.shape
        return data[slices]

    def set_data(self, values, slices=None, fill_others=False):
        if slices is None or slices2shape(slices) == self.shape:
            self.write_full(values.tobytes())
        else:
            if fill_others:
                cdata = np.full(self.shape, self.fillvalue, self.dtype)
            else:
                cdata = self.get_data()
            cdata[slices] = values
            self.write_full(cdata.tobytes())

    def write_full(self, data):

        self.client.put_object(
            Bucket=bucketName(self.dataset.name), Key=self.name, Body=data
        )

    def read(self, length=None, offset=0):
        if length is None:
            length = self.byte_count

        byteRange = 'bytes={}-{}'.format(offset, offset+length-1)
        return self.client.get_object(
            Bucket=bucketName(self.dataset.name),
            Key=self.name,
            Range=byteRange
        )['Body'].read()


_backend = Backend('s3', S3Connection, S3Dataset, S3DataChunk)
