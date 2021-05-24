#!/usr/bin/env python
"""backend RAM keeps every data structure in memory"""

import logging

import numpy as np

import random
import string

from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset, DatasetNotFoundError,
                                 BackendGroup, GroupNotFoundError)

log = logging.getLogger(__name__)
graph = {}
vertices_no = 0


class MemConnection(BackendConnection):
    """
    A Memory Connection represents a dictionary.
    """
    def __init__(self, *args, **kwargs):
        super(MemConnection, self).__init__(*args, **kwargs)
        self.root_group = MemGroup(self, "/")
        self.datasets = {}
        self.links = {}
        
    def create_group(self, path):
        if path != "/":
            return self.root_group.create_group(path)
        else:
            raise Exception("Group", path, "already exists")
        
    def get_group(self, path):
        return self.root_group.get_group(path)
    
    def has_group(self, path):
        return self.root_group.has_group(path)
        
    def del_group(self, path):
        self.root_group.del_group(path)
    
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

        log.debug('Creating Dataset `%s`', name)
        self.datasets[name] = None  # Key `name` has to exist
        dataset = MemDataset(self, name, shape, dtype, fillvalue, chunk_grid,
                             chunk_size)
        self.datasets[name] = dataset
        return dataset

    def get_dataset(self, name):
        if not self.has_dataset(name):
            raise DatasetNotFoundError('Dataset `%s` does not exist' % name)
        return self.datasets[name]

    def has_dataset(self, name):
        return name in self.datasets

    def del_dataset(self, name):
        if not self.has_dataset(name):
            raise DatasetNotFoundError('Dataset `%s` does not exist' % name)
        log.debug('Removing Dataset `%s`', name)
        del self.datasets[name]
        
class MemLink():
    def __init__(self, source, target, name):
        self.source = source
        self.target = target
        self.name = name
        
class MemGroup(BackendGroup):
    
    def __init__(self, parent, name, attrs=None, *args, **kwargs):
        super(MemGroup, self).__init__(name)
        self.parent = parent
        self.links = {}
        self.attrs = attrs
        self.datasets = {} # TODO
        self.connection = self.get_connection()
        self.absolute_path = self.get_absolute_path()
        
    def get_connection(self):
        
        def find_connection(parent):
            if parent.name == "/":
                return parent.parent.name
            else:
                return find_connection(parent.parent)
            
        if self.name == "/":
            return self.parent.name
        else:
            return find_connection(self.parent)
        
    def get_absolute_path(self):
        
        def find_path(parent):
            full_path = []
            if parent.name == "/":
                return full_path
            else:
                full_path.append(parent.name)
                full_path += find_path(parent.parent)
            return full_path
        
        if self.name == "/":
            return self.name
        else:
            full_path_list = find_path(self.parent)
            full_path_list.reverse()
            full_path_list.append(self.name)
            full_path = "/" + '/'.join(full_path_list)
            return full_path
    
    def keys(self):
        """
        Get the names of directly attached group memebers. 
        """
        return list(self.links.keys())
    
    def values(self):
        """
        Get the objects contained in the group (Group and Dataset instances).
        """
        objects = []
        for value in self.links.values():
            objects.append(value.target)
        return objects
    
    def items(self):
        """
        Get (name, value) pairs for object directly attached to this group.
        Values for broken soft or external links show up as None
        """
        items = {}
        for value in self.links.values():
            items[value.name] = value.target
        return items
    
    def create_group(self, path):
        """
        Creates a new empty group.
        :param string that provides an absolute path or a relative path to the new group
        """
        if not path.isalnum():
            raise Exception("String ", path, "is not alphanumeric")
        if not path in self.links:
            group = MemGroup(self, path)
            link = MemLink(self, group, path)
            self.links[path] = link
            
            return group

        else:
            raise Exception("Group", path, "already exists")
        
    def get_group(self, path):
        """
        Retrieve an item, or information about an item. work like the standard Python
        dict.get
        """
        def _recurse(arr, links):
            if arr[0] in links:
                link_target = links.get(arr[0]).target
                if len(arr) > 1:
                    arr.pop(0)
                    return _recurse(arr, link_target.links)
                else:
                    return link_target
        
        path_elements = path.split("/")
        group = _recurse(path_elements, self.links)
        
        if group is None:
            raise GroupNotFoundError("Group ", path, "does not exist")
        return group
    
    def has_group(self, path):
        """
        Return immediately attached groups to this group
        """
        def _recurse(arr, links):
            if arr[0] in links:
                link_target = links.get(arr[0]).target
                if len(arr) > 1:
                    arr.pop(0)
                    return _recurse(arr, link_target.links)
                else:
                    return link_target
        
        arr = path.split("/")
        group = _recurse(arr, self.links)
        
        if group is None:
            raise GroupNotFoundError("Group `%s` does not exist")
        return True
        
    
    def del_group(self, path):
        """
        Return immediately attached groups to this group
        """
        if not self.has_group(path):
            raise GroupNotFoundError("Group `%s` does not exist")

        # TODO remove group or link, or both?
        def _recurse(arr, links):
            if arr[0] in links:
                #link = links.get(arr[0])
                link_target = links.get(arr[0]).target
                log.debug('Removing Group `%s`', path)
                if len(arr) > 1:
                    arr.pop(0)
                    return _recurse(arr, link_target.links)
                else:
                    del links[arr[0]]
        
        arr = path.split("/")
        _recurse(arr, self.links)
        
    def visit(self):
        """
        Recursively visit all objects in this group and subgroups
        """
        def _recurse(links):
            groups = []
            for key, value in links.items():
                if hasattr(value.target, "links"):
                    groups.append(key)
                    groups += _recurse(value.target.links)
            return groups
        
        return _recurse(self.links)
    
    def visititems(self):
        """
        Recursively visit all objects in this group and subgroups.
        Like Group.visit(), except your callable should have the signature:
        callable (name, object)
        In this case object will be a Group ro Dataset instance
        """
        
        def _recurse(links):
            objects = []
            for key, value in links.items():
                objects.append(key)
                if hasattr(value.target, "links"):
                    objects += _recurse(value.target.links)
            return objects
        
        return _recurse(self.links)
        

    def create_dataset(
        self,
        name,
        shape=None,
        dtype=np.float32,
        fillvalue=0,
        data=None,
        chunk_size=None,
    ):

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
        self.datasets[name] = None  # Key `name` has to exist
        dataset = MemDataset(
            self, name, shape, dtype, fillvalue, chunk_grid, chunk_size,
        )
        self.datasets[name] = dataset
        
        link = MemLink(self, dataset, name)
        self.links[name] = link
        return dataset

    def get_dataset(self, name):
        if not self.has_dataset(name):
            raise DatasetNotFoundError("Dataset `%s` does not exist")
        return self.datasets[name]

    def has_dataset(self, name):
        return name in self.datasets

    def del_dataset(self, name):
        if not self.has_dataset(name):
            raise DatasetNotFoundError("Dataset `%s` does not exist")
        log.debug("Removing Dataset `%s`", name)
        del self.datasets[name]
        
    def create_metadata(self):
        return self.metadata
    
    def get_metadata(self):
        return self.metadata
    
    def has_metadata(self):
        return self.metadata
    
    def del_metadata(self):
        return self.metadata
    
    def get_object_info(self):
        """
        Get information about the group
        """
        object_info = {}
        object_info.update("Name", self.name)
        object_info.update("Type", "Node")
        object_info.update("Members", len(list(self.links.keys())))
        
        return object_info
        
    
class MemDataset(BackendDataset):

    def __init__(self, pool, name, shape, dtype, fillvalue, chunk_grid,
                 chunk_size):
        super(MemDataset, self).__init__(pool, name, shape, dtype, fillvalue,
                                         chunk_grid, chunk_size)
        self.data_chunks = {}
        self._populate_chunks()

    def _populate_chunks(self):
        for idx in np.ndindex(*self.chunk_grid):
            self.create_chunk(idx)

    def create_chunk(self, idx, data=None, slices=None):
        if self.has_chunk(idx):
            raise Exception('DataChunk `{}{}` already exists'.format(self.name,
                                                                     idx))

        self.data_chunks[idx] = None

        chunk = MemDataChunk(self, idx, 'Chunk {}'.format(idx),
                             self.chunk_size, self.dtype, self.fillvalue)
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


class MemDataChunk(BackendDataChunk):

    def __init__(self, dataset, idx, name, shape, dtype, fillvalue):
        super(MemDataChunk, self).__init__(dataset, idx, name, shape,
                                           dtype, fillvalue)
        self.data = np.full(shape, fillvalue, dtype)

    def get_data(self, slices=None):
        return self.data[slices]

    def set_data(self, values, slices=None):
        self.data[slices] = values


_backend = Backend('ram', MemConnection, MemDataset, MemDataChunk)
