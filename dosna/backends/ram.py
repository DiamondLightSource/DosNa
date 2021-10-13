#!/usr/bin/env python
"""backend RAM keeps every data structure in memory"""

import logging

import numpy as np

from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset, BackendGroup, BackendLink,
                                 DatasetNotFoundError, GroupNotFoundError)

log = logging.getLogger(__name__)


class MemConnection(BackendConnection):
    """
    A Memory Connection represents a dictionary.
    """
    def __init__(self, *args, **kwargs):
        super(MemConnection, self).__init__(*args, **kwargs)
        self.connection = self
        self.root_group = MemGroup(self, "/", attrs={})
        self.datasets = {}
        self.attrs = {}

    def keys(self):
        return self.root_group.keys()

    def values(self):
        return self.root_group.values()

    def items(self):
        return self.root_group.items()

    def create_group(self, path, attrs={}):
        if path == "/":
            raise Exception('Group: ', path, 'already exists')
        else:
            return self.root_group.create_group(path, attrs)

    def get_group(self, path):
        return self.root_group.get_group(path)

    def has_group(self, path):
        return self.root_group.has_group(path)

    def del_group(self, path):
        self.root_group.del_group(path)

    def get_object(self, path):
        return self.root_group.get_object(path)

    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunk_size=None, uuid=False):

        if not ((shape is not None and dtype is not None) or data is not None):
            raise Exception("Provide `shape` and `dtype` or `data`")
        if self.has_dataset(name):
            raise Exception("Dataset `%s` already exists" % name)

        if data is not None:
            shape = data.shape
            dtype = data.dtype

        if chunk_size is None:
            chunk_size = shape

        chunk_grid = (np.ceil(np.asarray(shape, float) / chunk_size)).astype(int)

        log.debug("Creating Dataset `%s`", name)
        self.datasets[name] = None

        dataset = MemDataset(self, name, shape, dtype, fillvalue, chunk_grid, chunk_size)

        self.datasets[name] = dataset

        return dataset

    def get_dataset(self, name):
        if not self.has_dataset(name):
            raise DatasetNotFoundError("Dataset `%s` does not exist" % name)
        return self.datasets[name]

    def has_dataset(self, name):
        return name in self.datasets

    def del_dataset(self, name):
        if not self.has_dataset(name):
            raise DatasetNotFoundError("Dataset `%s` does not exist" % name)
        log.debug("Removing Dataset `%s`", name)
        del self.datasets[name]


    def visit_objects(self):
        return self.root_group.visit_objects()

class MemLink(BackendLink):
    def __init__(self, source, target, name):
        super(MemLink, self).__init__(source, target, name)

    def get_source(self):
        return self.source

    def get_target(self):
        return self.target

    def get_name(self):
        return self.name

class MemGroup(BackendGroup):
    def __init__(self, parent, name, attrs={}, path_split="/", *args, **kwargs):
        super(MemGroup, self).__init__(parent, name, attrs)

        self.attrs = attrs
        self.links = {}
        self.datasets = {}
        self.path_split = path_split
        self.connection = parent.connection
        self.absolute_path = self.get_absolute_path()

    def get_absolute_path(self):

        def _find_path(group):
            full_path = []
            if group.name == "/":
                return full_path
            else:
                full_path.append(group.name)
                full_path += _find_path(group.parent)
            return full_path

        full_path_list = _find_path(self)
        full_path_list.reverse()
        full_path = "/" + self.path_split.join(full_path_list)

        return full_path

    def keys(self):
        return list(self.links.keys())

    def values(self):
        objects = []
        for value in self.links.values():
            objects.append(value.target)
        return objects

    def items(self):
        items = dict()
        for value in self.links.values():
            items[value.name] = value.target
        return items

    def create_group(self, path, attrs={}):

        def _create_subgroups(path, group):
            subgroup = MemGroup(group, path[0])
            link = MemLink(group, subgroup, path[0])
            group.links[path[0]] = link
            path.pop(0)
            if len(path) == 0:
                return False
            _create_subgroups(path, subgroup)

        if path in self.links:
            raise Exception("Group", path, "already exists")

        elif self.path_split in path:
            path_elements = path.split(self.path_split)
            _create_subgroups(path_elements, self)

        else:
            group = MemGroup(self, path, attrs)
            link = MemLink(self, group, path)
            self.links[path] = link
            return group


    def get_group(self, path):

        def _find_group(path, links):
            first_element = path[0]
            if first_element in links:
                object = links.get(first_element).target
                if len(path) > 1:
                    path.pop(0)
                    return _find_group(path, object.links)
                else:
                    if hasattr(object, 'links'):
                        return object

        path_elements = path.split(self.path_split)

        if path.startswith(self.path_split):
            path_elements.pop(0)
            links = self.connection.root_group.get_links()
        else:
            links = self.links

        group = _find_group(path_elements, links)

        if group is None:
            raise GroupNotFoundError('Group', path, 'not found.')
        else:
            return group

    def get_object(self, path):

        def _find_object(path, links):
            first_element = path[0]
            if first_element in links:
                object = links.get(first_element).target
                if len(path) > 1:
                    path.pop(0)
                    return _find_object(path, object.links)
                else:
                    return object

        path_elements = path.split(self.path_split)

        if path.startswith(self.path_split):
            path_elements.pop(0)
            links = self.connection.root_group.get_links()
        else:
            links = self.links

        object = _find_object(path_elements, links)

        if object is None:
            raise Exception('Object', path, 'not found')
        else:
            return object

    def has_group(self, path):
        try:
            valid = self.get_group(path)
        except GroupNotFoundError:
            return False
        return valid

    def del_group(self, path):
        if not self.has_group(path):
            raise GroupNotFoundError('Group', path, 'does not exist.')

        def _del_link(path, links):
            if path[0] in links:
                subgroup = links.get(path[0]).target
                log.debug('Removing group: ', path)
                if len(path) > 1:
                    path.pop(0)
                    return _del_link(path, subgroup.links)
                else:
                    del subgroup
                    del links[path[0]]

        path_elements = path.split(self.path_split)
        _del_link(path_elements, self.links)

    def visit_groups(self):

        def _recurse(links):
            groups = []
            for key, value in links.items():
                subgroup = value.target
                if hasattr(subgroup, 'links'):
                    groups.append(subgroup.get_absolute_path())
                    groups += _recurse(subgroup.links)
            return groups

        return _recurse(self.links)

    def visit_objects(self):

        def _recurse(links):
            objects = []
            for key, value in links.items():
                objects.append(value.target.get_absolute_path())
                if hasattr(value.target, 'links'):
                    objects += _recurse(value.target.links)
            return objects

        return _recurse(self.links)

    def add_metadata(self, attrs):
        self.attrs.update(attrs)
        return self.attrs

    def get_metadata(self):
        return self.attrs

    def has_metadata(self):
        if self.attrs:
            return self.attrs

    def del_metadata(self):
        self.attrs.clear()

    def create_dataset(self,name,shape=None,dtype=np.float32,fillvalue=0,
                       data=None,chunk_size=None,uuid=False,):

        if not ((shape is not None and dtype is not None) or data is not None):
            raise Exception("Provide `shape` and `dtype` or `data`")
        if self.has_dataset(name):
            raise Exception("Dataset `%s` already exists" % name)

        if data is not None:
            shape = data.shape
            dtype = data.dtype

        if chunk_size is None:
            chunk_size = shape

        chunk_grid = (np.ceil(np.asarray(shape, float) / chunk_size)).astype(int)

        log.debug("Creating Dataset `%s`", name)
        self.datasets[name] = None
        dataset = MemDataset(self, name, shape, dtype, fillvalue,
                             chunk_grid, chunk_size)
        self.datasets[name] = dataset

        link = MemLink(self, dataset, name)
        self.links[name] = link
        return dataset

    def get_dataset(self, path):

        def _find_group(path, links):
            first_element = path[0]
            if first_element in links:
                object = links.get(first_element).target
                if len(path) > 1:
                    path.pop(0)
                    return _find_group(path, object.links)
                else:
                        return object

        path_elements = path.split(self.path_split)
        if path.startswith("/"):
            path_elements.pop(0)
            links = self.connection.root_group.get_links()
        else:
            links = self.links

        group = _find_group(path_elements, links)

        if group is None:
            raise DatasetNotFoundError("Dataset", path, "not found")
        else:
            return group

    def has_dataset(self, name):
        return name in self.datasets

    def del_dataset(self, name):
        if not self.has_dataset(name):
            raise DatasetNotFoundError("Dataset `%s` does not exist")
        log.debug("Removing Dataset `%s`", name)
        del self.datasets[name]

class MemDataset(BackendDataset):
    def __init__(self, pool, name, shape, dtype, fillvalue, chunk_grid,
                 chunk_size, path_split="/"):

        super(MemDataset, self).__init__(pool, name, shape, dtype, fillvalue,
                                         chunk_grid, chunk_size)
        self.data_chunks = {}
        self._populate_chunks()

        self.parent = pool
        self.path_split = path_split

    def _populate_chunks(self):
        for idx in np.ndindex(*self.chunk_grid):
            self.create_chunk(idx)

    def create_chunk(self, idx, data=None, slices=None):
        if self.has_chunk(idx):
            raise Exception("DataChunk `{}{}` already exists".format(self.name, idx))

        self.data_chunks[idx] = None

        chunk = MemDataChunk(
            self,
            idx,
            "Chunk {}".format(idx),
            self.chunk_size,
            self.dtype,
            self.fillvalue,
        )
        if data is not None:
            slices = slices or slice(None)
            chunk.set_data(data, slices=slices)

        self.data_chunks[idx] = chunk
        return chunk

    def get_chunk(self, idx):
        if self.has_chunk(idx):
            return self.data_chunks[idx]
        return self.create_chunk(idx)

    def has_chunk(self, idx):
        return idx in self.data_chunks

    def del_chunk(self, idx):
        if self.has_chunk(idx):
            del self.data_chunks[idx]
            return True
        return False

    def get_absolute_path(self):

        def find_path(group):
            full_path = []
            if group.name == "/":
                return full_path
            else:
                full_path.append(group.name)
                full_path += find_path(group.parent)
            return full_path

        full_path_list = find_path(self)
        full_path_list.reverse()
        full_path = "/" + self.path_split.join(full_path_list)
        return full_path

class MemDataChunk(BackendDataChunk):
    def __init__(self, dataset, idx, name, shape, dtype, fillvalue):
        super(MemDataChunk, self).__init__(dataset, idx, name, shape, dtype, fillvalue)
        self.data = np.full(shape, fillvalue, dtype)

    def get_data(self, slices=None):
        return self.data[slices]

    def set_data(self, values, slices=None):
        self.data[slices] = values


_backend = Backend("ram", MemConnection, MemDataset, MemDataChunk)
