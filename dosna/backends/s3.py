
import logging

import boto3
import numpy as np

from botocore.exceptions import ClientError
from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset, BackendGroup, BackendLink,
                                 ConnectionError, DatasetNotFoundError,
                                 GroupNotFoundError)

from dosna.util import dtype2str, shape2str, str2shape
from dosna.util.data import slices2shape

_DATASET_ROOT = 'dataset_root'
_SIGNATURE = 'DosNa Dataset'

_PATH_SPLIT = '/'
_ROOT_GROUP = '/'

_SHAPE = 'shape'
_DTYPE = 'dtype'
_FILLVALUE = 'fillvalue'
_CHUNK_GRID = 'chunk-grid'
_CHUNK_SIZE = 'chunk-size'

log = logging.getLogger(__name__)

class S3Connection(BackendConnection):

    def __init__(self, name, bucket, endpoint_url, verify=True,
                 profile_name='default', region_name='us-west-2',
                 aws_access_key=None, aws_secret_access_key=None,
                 *args, **kwargs):

        super(S3Connection, self).__init__(name, *args, **kwargs)

        self._endpoint_url = endpoint_url
        self._bucket = bucket
        self._verify = verify
        self._profile_name = profile_name
        self._region_name = region_name
        self._aws_access_key = aws_access_key
        self._aws_secret_access_key = aws_secret_access_key

        #self._root_group = S3Group(self,_ROOT_GROUP,attrs={})
        self.groups = {}
        self.datasets = {}


        super(S3Connection, self).__init__(name, *args, **kwargs)

    def connect(self):

        if self.connected:
            raise ConnectionError(
                'Connection {} is already open'.format(self.name))

        session = boto3.session.Session(profile_name=self._profile_name)

        self._client = boto3.client('s3',
                                    endpoint_url=self._endpoint_url,
                                    verify=self._verify,
                                    region_name=self._region_name,
                                    aws_access_key_id=self._aws_access_key,
                                    aws_secret_access_key=self._aws_secret_access_key,
                                    )
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError as e:
            code = e.response['Error']['Code']
            if code is not None:
                log.error('connect: head_bucket returns %s', code)
                return None

        super(S3Connection, self).connect()

    def disconnect(self):

        if self.connected:
            super(S3Connection, self).disconnect()

    @property
    def bucket(self):
        return self._bucket

    @property
    def client(self):
        return self._client

    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunk_size=None):

        absolute_path = self.name + _PATH_SPLIT + name

        if not ((shape is not None and dtype is not None) or data is not None):
            raise Exception('Provide `shape` and `dtype` or `data`')
        #if self.has_dataset(name): # TODO
        #    raise Exception('Dataset `%s` already exists' % absolute_path)

        if data is not None:
            shape = data.shape
            dtype = data.dtype

        if chunk_size is None:
            chunk_size = shape

        chunk_grid = (np.ceil(np.asarray(shape, float) / chunk_size)).astype(int)

        metadata = {
            _SHAPE: shape2str(shape),
            _DTYPE: dtype2str(dtype),
            _FILLVALUE: repr(fillvalue),
            _CHUNK_GRID: shape2str(chunk_grid),
            _CHUNK_SIZE: shape2str(chunk_size)
        }

        self._client.put_object(
            Bucket=self._bucket,
            Key=absolute_path,
            Body=str.encode(_SIGNATURE),
            Metadata=metadata
        )

        self.datasets[absolute_path] = None

        dataset = S3Dataset(
            self, absolute_path, shape, dtype,
            fillvalue, chunk_grid, chunk_size
        )

        self.datasets[absolute_path] = dataset

        return dataset

    def create_group(self, name, attrs={}):

        keys_values = attrs.items()

        attrs = {str(key): str(value) for key, value in keys_values}

        absolute_path = self.name + _PATH_SPLIT + name

        self.client.put_object(Bucket=self.bucket, Key=absolute_path,
                               Body=str.encode('Group')) #Metadata=attrs)

        group = S3Group(self, absolute_path, attrs)

        self.groups[absolute_path] = group

        return group

    def get_group(self, name):

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=name)['Body'].read()
            print(response)
        except Exception as e:
            print('Group: ', name, 'does not exist. ', e)

        print('selfgroups', self.groups)
        return response

    def get_dataset(self, name):

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=name)['Body'].read()
            print(response)
        except Exception as e:
            print('Group: ', name, 'does not exist. ', e)

        return self.datasets[name]

    def visit_objects(self):

        prefix = self.name

        files = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        objects_list = []
        for f in files['Contents']:
            objects_list.append(f['Key'])

        return objects_list


class S3Group(BackendGroup):

    @property
    def client(self):
        return self.parent.client

    @property
    def bucket(self):
        return self.parent.bucket


    def create_group(self, name, attrs={}):

        keys_values = attrs.items()

        attrs = {str(key): str(value) for key, value in keys_values}

        absolute_path = self.name + _PATH_SPLIT + name

        self.client.put_object(Bucket=self.bucket, Key=absolute_path,
                               Body=str.encode('Group')) #Metadata=attrs)

        group = S3Group(self, absolute_path, attrs)

        return group


    def get_group(self, name):

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=name)['Body'].read()
        except Exception as e:
            print('Group: ', name, 'does not exist. ', e)

        return response # TODO return S3GROUP


    def has_group(self, name):
        try:
            self.client.head_object(Bucket=self.bucket, Key=name)
            return True
        except Exception as e:
            print('Group not found: ', name, 'Error: ',  e)
            return False

    def del_group(self, name):

        prefix = name

        if not self.has_group(name):
            raise Exception('Object: ', name, 'does not exist')
        else:
            files = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            for f in files['Contents']:
                response = self.client.delete_object(Bucket=self.bucket, Key=f['Key'])
                if response is not None:
                    print('Deleting', f['Key'])
                else:
                    print('Could not delete object', f[key])


    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunk_size=None):

        if not ((shape is not None and dtype is not None) or data is not None):
            raise Exception('Provide `shape` and `dtype` or `data`')
        # TODO restore this but with the appropiate function
#        if self.has_dataset(name):
#            raise Exception('Dataset `%s` already exists' % name)

        if data is not None:
            shape = data.shape
            dtype = data.dtype

        if chunk_size is None:
            chunk_size = shape

        chunk_grid = (np.ceil(np.asarray(shape, float) / chunk_size)) \
            .astype(int)

        metadata = {
            _SHAPE: shape2str(shape),
            _DTYPE: dtype2str(dtype),
            _FILLVALUE: repr(fillvalue),
            _CHUNK_GRID: shape2str(chunk_grid),
            _CHUNK_SIZE: shape2str(chunk_size)
        }

        absolute_path = self.name + _PATH_SPLIT + name

        self.client.put_object(Bucket=self.bucket,Key=absolute_path,
                               Body=str.encode(_SIGNATURE),Metadata=metadata)

        # TODO NAME of dataset
        dataset = S3Dataset(self, absolute_path, shape, dtype,fillvalue, chunk_grid, chunk_size)

        return dataset

    def get_dataset(self, name):

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=name)['Body'].read()
        except Exception as e:
            print('Group: ', name, 'does not exist. ', e)

        return response  # TODO return S3GROUP

    def has_dataset(self, name):
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=name)
        except Exception as e:
            print('Dataset: ', name, 'not found')

    def del_dataset(self, name):
        pass


class S3Dataset(BackendDataset):

    @property
    def client(self):
        return self.connection.client

    @property
    def bucket(self):
        return self.connection.bucket

    def _idx2name(self, idx):
        return '.'.join(map(str, idx))

    def create_chunk(self, idx, data=None, slices=None):
        if self.has_chunk(idx):
            raise Exception('DataChunk `{}{}` already exists'.
                            format(self.name, idx))
        name = self._idx2name(idx)
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
        absolute_path = self.name + _PATH_SPLIT + name

        try:
            self.client.head_object(Bucket=self.bucket, Key=absolute_path)
            has_chunk = True
        except ClientError as e:
            log.debug("ClientError: %s", e.response['Error']['Code'])

        return has_chunk

    def del_chunk(self, idx):
        name = self._idx2name(idx)
        absolute_path = self.name + _PATH_SPLIT + name
        if self.has_chunk(idx):
            self.client.delete_object(
                Bucket=self.bucket,
                Key=absolute_path
            )
            return True
        return False

class S3DataChunk(BackendDataChunk):

    @property
    def client(self):
        return self.dataset.client

    @property
    def bucket(self):
        return self.dataset.bucket


    def get_data(self, slices=None):
        if slices is None:
            slices = slice(None)

        data = np.fromstring(self.read(), dtype=self.dtype, count=self.size)
        data.shape = self.shape
        return data[slices]

    def set_data(self, values, slices=None, fill_others=False):
        if slices is None or slices2shape(slices) == self.shape:
            if not isinstance(values, bytes):
                self.write_full(values.tobytes())
        else:
            if fill_others:
                cdata = np.full(self.shape, self.fillvalue, self.dtype)
            else:
                cdata = self.get_data()
            cdata[slices] = values
            self.write_full(cdata.tobytes())

    def write_full(self, data):
        absolute_path = self.dataset.name + "/" + self.name
        print('Writing data_chunk')
        self.client.put_object(
            Bucket=self.dataset.bucket, Key=absolute_path, Body=data, # TODO check if tagging works
        )

    def read(self, length=None, offset=0):
        if length is None:
            length = self.byte_count
        absolute_path = self.dataset.name + "/" + self.name
        byteRange = 'bytes={}-{}'.format(offset, offset+length-1)
        print('Retrieving data chunk')
        return self.client.get_object(
            Bucket=self.dataset.bucket,
            Key=absolute_path,
            Range=byteRange
        )['Body'].read()


_backend = Backend('s3', S3Connection, S3Dataset, S3DataChunk)


