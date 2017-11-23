#!/usr/bin/env python
"""Base classes for every engine"""


import logging

import numpy as np

log = logging.getLogger(__name__)


class BackendWrapper(object):

    instance = None

    def __init__(self, instance):
        self.instance = instance

    def __getattr__(self, attr):
        """
        Attributes/Functions that do not exist in the extended class
        are going to be passed to the instance being wrapped
        """
        return self.instance.__getattribute__(attr)

    def __enter__(self):
        self.instance.__enter__()
        return self

    def __exit__(self, *args):
        self.instance.__exit__(*args)


class EngineConnection(BackendWrapper):

    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunk_size=None):
        self.instance.create_dataset(name, shape, dtype, fillvalue,
                                     data, chunk_size)
        dataset = self.get_dataset(name)
        if data is not None:
            dataset.load(data)
        return dataset

    def get_dataset(self, name):
        # this is meant to wrap the dataset with the specific engine class
        raise NotImplementedError('`get_dataset` not implemented '
                                  'for this engine')

    def del_dataset(self, name):
        dataset = self.get_dataset(name)
        dataset.clear()
        self.instance.del_dataset(name)

    def __getitem__(self, dataset_name):
        return self.get_dataset(dataset_name)


class EngineDataset(BackendWrapper):

    def get_data(self, slices=None):
        raise NotImplementedError('`get_data` not implemented for this engine')

    def set_data(self, values, slices=None):
        raise NotImplementedError('`set_data` not implemented for this engine')

    def clear(self):
        raise NotImplementedError('`clear` not implemented for this engine')

    def delete(self):
        self.clear()
        self.instance.connection.del_dataset(self.name)

    def load(self, data):
        raise NotImplementedError('`load` not implemented for this engine')

    def map(self, func, output_name):
        raise NotImplementedError('`map` not implemented for this engine')

    def apply(self, func):
        raise NotImplementedError('`apply` not implemented for this engine')

    def clone(self, output_name):
        raise NotImplementedError('`clone` not implemented for this engine')

    def create_chunk(self, idx, data=None, slices=None):
        self.instance.create_chunk(idx, data, slices)
        return self.get_chunk(idx)

    def get_chunk(self, idx):
        raise NotImplementedError('`create_chunk` not implemented '
                                  'for this backend')

    def del_chunk(self, idx):
        # just for base completeness
        self.instance.del_chunk(idx)

    def __getitem__(self, slices):
        return self.get_data(slices)

    def __setitem__(self, slices, values):
        self.set_data(values, slices=slices)


class EngineDataChunk(BackendWrapper):
    pass
