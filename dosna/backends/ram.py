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
        self.datasets = {}
        self.links = {}
        
    def create_group(self, path):
        """
        Creates a new empty group.
        :param string that provides an absolute path or a relative path to the new group
        """
        if not path in self.links:
            group = MemGroup(self, path)
            link = MemLink(self, group, path)
            self.links[path] = link
            
            return group
        else:
            raise Exception("Group", path, "already exists")
            
    
    def get_group(self, path):
        """
        Return immediately attached groups to this group
        """
        if not self.has_group(path):
            raise GroupNotFoundError("Group `%s` does not exist")
        return self.links[path].target
    
    def has_group(self, path):
        """
        Return immediately attached groups to this group
        """
        return path in self.links
    
    def del_group(self, path):
        """
        Return immediately attached groups to this group
        """
        if not self.has_group(path):
            raise GroupNotFoundError("Group `%s` does not exist")
        log.debug('Removing Group `%s`', path)
        del self.links[path].name
    
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
    
    def __init__(self, name, *args, **kwargs):
        
        super(MemGroup, self).__init__(name)
        #self.name = name
        #self.parent = parent
        # self.connection = file
        self.links = {}
        self.attrs = {}
        self.datasets = {}
    
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
        #ValuesViewHDF5(<HDF5 group "/Bazz" (3 members)>)
        return objects # TODO: is this the actual object? this seems the same
    
    def items(self):
        """
        Get (name, value) pairs for object directly attached to this group.
        Values for broken soft or external links show up as None
        """
        items = {}
        for value in self.links.values():
            items[value.name] = value.target
        return items # TODO: implement this
    
    def create_group(self, path):
        """
        Creates a new empty group.
        :param string that provides an absolute path or a relative path to the new group
        """
        
        if not path in self.links:
            
            group = MemGroup(self, path)
            link = MemLink(self, group, path)
            self.links[path] = link
            
            return group

        else:
            raise Exception("Group", path, "already exists")
        
    def get_group(self, path):
        """
        Return immediately attached groups to this group
        """
        if not self.has_group(path):
            raise GroupNotFoundError("Group `%s` does not exist")
        return self.links[path].target
    
    def has_group(self, path):
        """
        Return immediately attached groups to this group
        """
        return path in self.links
    
    def del_group(self, path):
        """
        Return immediately attached groups to this group
        """
        if not self.has_group(path):
            raise GroupNotFoundError("Group `%s` does not exist")
        log.debug('Removing Group `%s`', path)
        del self.links[path]
        
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
        
        
    def get_node(self, path): #TODO regex
        """
        Retrieve an item, or information about an item. work like the standard Python
        dict.get
        """
        if path in self.links:
            return self.links[path]
        elif "/" in path:
            arr = path.split("/")
            """
                    
            if arr[0] in self.links:
                first_link = self.links.get(arr[0])
                if arr[1] in first_link.target.links:
                    second_link = first_link.target.links.get(arr[1])
                    if arr[2] in second_link.target.links:
                        third_link = second_link.target.links.get(arr[2])
                        return third_link.target
            """
        
    def iterate(self): # TODO: allow for cycles
        """
        Recursively visits all the objects.
        Structure is traversed as a graph, starting at one node
        and recursively visiting linked nodes. 
        """
        global graph
        visited = [] 
        def dfs(visited, graph, node):
            if node not in visited:
                visited.append(node)
                for neighbour in graph[node]:
                    dfs(visited, graph, neighbour)
        dfs(visited, graph, self.name)
        return visited
    


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
            self, name, shape, dtype, fillvalue, chunk_grid, chunk_size
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
    
    def get_object_info(self): # TODO: in the metadata?
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
