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
        global graph
        super(MemConnection, self).__init__(*args, **kwargs)
        self.datasets = {}
        self.trees = {}
        
    def create_node(self, location, path):
        """
        Creates a new empty group and gives it a name
        :param location: identifier of the file/group in a file with respect to which the new group is to be identified
        :param path: string that provides wither an absolute path or a relative path to the new group
                     Begins with a slash: absolute path indicating that it locates the new group from the root group of the HDF5 file.
                     No slash: relative path is a path from that file's root group.
                               when the location is a group, a relative path is a path from that group.
        """
        global graph
        global vertices_no
        
        if path.startswith("/"):
            log.info("Absolute path")
        else:
            log.info("Relative path")
            
        
        if path in graph:
            raise Exception("Group", path, "already exists")
        else:
            vertices_no = vertices_no + 1
            
            graph[path] = []
            
            node = MemGroup(self, path)
            edge = random.choice(string.ascii_letters)
            #self.link(self.name, path, edge)
            
            return node
    
    def link(self, n1, n2, e):
        global graph
        if n1 not in graph:
            raise Exception("Node", n1, "does not exist")
        if n2 not in graph:
            raise Exception("Node", n2, "does not exist")
        else:
            temp = [n2, e] # TODO: what is this?
            graph[n1].append(e) #TODO: append e or the n2?
        return e
        
    def create_tree(self, name):
        self.trees[name] = {}
        backendtree = MemTree(self, name)
        self.trees[name][name] = backendtree
        return backendtree
    
    def get_tree(self, name):
        if not self.has_tree(name):
            raise BackendTreeNotFoundError("Backend `%s` does not exist")
        return self.trees[name][name]

    def has_tree(self, name):
        return name in self.trees

    def del_tree(self, name):
        if not self.has_dataset(name):
            print("BackendTree not found") # TODO: Implement class
        log.debug("Removing Backend `%s`", name)
        del self.trees[name]

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
    def __init__(self, name, source, target):
        self.name = name
        self.source = source
        self.target = target
        
class MemGroup(BackendGroup):
    
    def __init__(self, parent, name, *args, **kwargs):
        super(MemGroup, self).__init__(name)
        #self.name = name # TODO: full path to this group
        self.parent = parent
        # self.file = file # TODO: file instance in which this group resides
        self.links = {}
        self.attrs = {}
        self.no_members = 0
    
    def __contains__(self, path):
        if path in self.links:
            return True
        else:
            return False
    
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
    
    def create_node(self, path):
        """
        Creates a new empty group and gives it a name
        """
        global graph
        global vertices_no
        
        #if path.startswith("/"):
        #    log.info("Absolute path")
        #else:
        #    log.info("Relative path"
        
        if path in graph and path in self.links:
            raise Exception("Group", path, "already exists")
        else:
            graph[path] = []
            graph[self.name].append(path)
            vertices_no = vertices_no + 1
            
            node = MemGroup(self, path)
            link = MemLink(path, self, node)
            self.links[path] = link

            return node
        
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
    
    def get_object_info(self):
        """
        Get information about the group
        """
        object_info = {}
        object_info.update("Name", self.name)
        object_info.update("Type", "Node")
        object_info.update("Members", len(list(self.links.keys())))
        
        return object_info
    
    def get_link_info():
        return self.links
    
    def get_info(self):
        return self.metadata
    
    def link(self, n1, n2, e): # TODO: not sure if leaving it
        global graph
        if n1 not in graph:
            raise Exception("Node", n1, "does not exist")
        if n2 not in graph:
            raise Exception("Node", n2, "does not exist")
        else:
            graph[n1].append(e)
        return e
    
    def unlink(self, n1, n2, e): # TODO: not sure if leaving it
        global graph
        if n1 not in graph:
            raise Exception("Node", n1, "does not exist")
        if n2 not in graph:
            raise Exception("Node", n2, "does not exist")
        else:
            graph[n1].remove(e)
        return e

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
        
    
        
    
        
class MemTree(): # TODO: add the BackendTree
    """
    A Memory Tree represents a dictionary of dictionaries
    """
    
    def __init__(self, connection_handler, name, *args, **kwargs):
        # super(MemTree, self).__init__(*args, **kwargs)
        self.connection_handler = connection_handler
        self.name = name
        self.metadata = {}
        self.datasets = {}
        self.trees = {}
        self.graph = {}
        self.graph[self.name] = []
        
    # Added methods
    def create(self, location, path):
        """
        Creates a new empty group and gives it a name
        :param location: identifier of the file/group in a file with respect to which the new group is to be identified
        :param path: string that provides wither an absolute path or a relative path to the new group
                     Begins with a slash: absolute path indicating that it locates the new group from the root group of the HDF5 file.
                     No slash: relative path is a path from that file's root group.
                               when the location is a group, a relative path is a path from that group.
        """
        if path.startswith("/"):
            pass
        else:
            pass
        tree = MemTree(self, path)
        self.trees[path] = tree
        return tree
    
    def create_tree(self, name):
        self.trees[name] = {}
        backendtree = MemTree(self, name)
        self.trees[name][name] = backendtree
        return backendtree
    
    def open(self, name):
        """ Open an existing group"""
        tree = self.trees.get(name, None)
        return tree

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
