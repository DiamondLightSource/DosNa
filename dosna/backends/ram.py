#!/usr/bin/env python
"""backend RAM keeps every data structure in memory"""

import logging

import numpy as np

import random
import string

from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset, DatasetNotFoundError,
                                 BackendGroup, GroupNotFoundError,
                                 BackendLink)

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

    def create_group(self, path, attrs={}):
        if not path.isalnum():
            raise Exception("String ", path, "is not alphanumeric")
        if path != "/":
            return self.root_group.create_group(path, attrs)
        else:
            raise Exception("Group", path, "already exists")
        
    def get_group(self, path):
        return self.root_group.get_group(path)
    
    def has_group(self, path):
        return self.root_group.has_group(path)
        
    def del_group(self, path):
        self.root_group.del_group(path)

    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunk_size=None, uuid=False):
        """
        if uuid:
            name = name = name + "-" + str(uuid.uuid4())
        """
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

        self.parent = parent
        self.path_split = path_split
        self.connection = parent.connection

        self.attrs = attrs
        self.links = {}
        self.datasets = {}

    """
    def get_connection(self):
        Recursively access the parent groups until the parent group is the
        root group which is "/", then get the parent name of the root group
        which is the name of the connection.
        
        :return name of the DosNa connection

        def find_connection(group):
            if group.name == "/":
                return group.parent.name
            else:
                return find_connection(group.parent)
            
        return find_connection(self)
    """
    def get_absolute_path(self):
        """
        Recursively access the parent groups until the parent name is the root group "/"
        and append the name of the parent groups to obtain the full path from the root group.
        
        :return absolute path name from the root group
        """
        
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
        Get (name, value) pairs for objects directly attached to this group.
        """
        items = {}
        for value in self.links.values():
            items[value.name] = value.target
        return items
    
    def create_group(self, path, attrs={}):
        """
        Creates a new empty group.
        Validates the path is alphanumeric.
        If path is not in the links attached to the group, it will create a new group and link.
        The link will the current group as source and the new group as target. The name of the link
        is the name of the group.
        :param string that provides an absolute path or a relative path to the new group
        :return new group
        """
        if '/' in path:
            path_elements = path.split('/')
            print(path_elements)
            def recurse(path, group):
                subgroup = MemGroup(group, path[0])
                link = MemLink(group, subgroup, path[0])
                group.links[path[0]] = link
                path.pop(0)
                if len(path) == 0:
                    return False
                recurse(path, subgroup)


            recurse(path_elements, self)
            print(self.links)

        #if not path.isalnum():
        #    raise Exception("String ", path, "is not alphanumeric")

        elif path in self.links:
            raise Exception("Group", path, "already exists")
        else:
            group = MemGroup(self, path, attrs)
            link = MemLink(self, group, path)
            self.links[path] = link
            return group
        """
        if not path.isalnum():
            raise Exception("String ", path, "is not alphanumeric")
        elif path in self.links:
            raise Exception("Group", path, "already exists")
        else:
            group = MemGroup(self, path, attrs)
            link = MemLink(self, group, path)
            self.links[path] = link
            return group
        """
            
        
    def get_group(self, path):
        """
        Splits the path string for each slash found.
        For each element in the resulting array, it checks recursively whether the first element
        of the array is in the dictionary of links. If it is, it pops the the first element and
        performs the same process with the next element of the array and the next group links.
        
        :param string that provides an absolute path or a relative path to the new group
        :return DosNa group
        """

        def _recurse(path, links):
            first_element = path[0]
            if first_element in links:
                object = links.get(first_element).target
                if len(path) > 1:
                    path.pop(0)
                    return _recurse(path, object.links)
                else:
                    if hasattr(object, 'links'):
                        return object

        path_elements = path.split(self.path_split)
        if path.startswith("/"):
            path_elements.pop(0)
            links = self.connection.root_group.get_links()
        else:
            links = self.links

        group = _recurse(path_elements, links)

        if group is None:
            raise GroupNotFoundError("Group", path, "not found")
        else:
            return group
    
    def has_group(self, path):
        """
        Splits the path string for each slash found.
        For each element in the resulting array, it checks recursively whether the first element
        of the array is in the dictionary of links. If it is, it pops the the first element and
        performs the same process with the next element of the array and the next group links.
        """
        if self.get_group(path):
            return True
        else:
            raise GroupNotFoundError("Group", path, "does not exist")
        
    
    def del_group(self, path):
        """
        Recursively access links to find group, and then deletes it. 
        """
        if not self.has_group(path):
            raise GroupNotFoundError("Group", path, "does not exist")

        def _recurse(path, links):
            if path[0] in links:
                subgroup = links.get(path[0]).target
                log.debug("Removing Group", path)
                if len(path) > 1:
                    path.pop(0)
                    return _recurse(path, subgroup.links)
                else:
                    del subgroup
                    del links[path[0]]
        
        path_elements = path.split(self.path_split)
        _recurse(path_elements, self.links)
        
    def visit_groups(self):
        """
        Recursively visit all objects in this group and subgroups
        :return all objects names of the groups and subgroups of this group
        """
        def _recurse(links):
            groups = []
            for key, value in links.items():
                subgroup = value.target
                if hasattr(subgroup, "links"):
                    groups.append(subgroup.get_absolute_path())
                    groups += _recurse(subgroup.links)
            return groups
        
        return _recurse(self.links)
    
    def visit_objects(self):
        """
        Recursively visit all objects in this group and subgroups
        :return all objects names of the groups, subgroups and datasets of this group
        """
        
        def _recurse(links):
            objects = []
            for key, value in links.items():
                objects.append(value.target.get_absolute_path())
                if hasattr(value.target, "links"):
                    objects += _recurse(value.target.links)
            return objects
        
        return _recurse(self.links)
    """
    def get_object(self, path):
        Splits the path string for each slash found.
        For each element in the resulting array, it checks recursively whether the first element
        of the array is in the dictionary of links. If it is, it pops the the first element and
        performs the same process with the next element of the array and the next group links.

        :param string that provides an absolute path or a relative path to the new group
        :return DosNa group
        if path.startswith("/"):  
            raise GroupNotFoundError("Group ", path, "does not exist")

        def _recurse(path, links):
            first_element = path[0]
            if first_element in links:
                object = links.get(first_element).target
                if len(path) > 1:
                    path.pop(0)
                    return _recurse(path, object.links)
                else:
                    return object

        path_elements = path.split(self.path_split)
        object = _recurse(path_elements, self.links)
        if object is None:
            raise GroupNotFoundError("Group", path, "not found")
        else:
            return object
    """

    def get_links(self):
        return self.links
        
    def create_dataset(
        self,
        name,
        shape=None,
        dtype=np.float32,
        fillvalue=0,
        data=None,
        chunk_size=None,
        uuid=False
    ):
        """
        if uuid:
            name = name = name + "-" + str(uuid.uuid4())
        """
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

    def get_absolute_path(self):
        """
        Recursively access the parent groups until the parent name is the root group "/"
        and append the name of the parent groups to obtain the full path from the root group.

        :return absolute path name from the root group
        """

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
        super(MemDataChunk, self).__init__(dataset, idx, name, shape,
                                           dtype, fillvalue)
        self.data = np.full(shape, fillvalue, dtype)

    def get_data(self, slices=None):
        return self.data[slices]

    def set_data(self, values, slices=None):
        self.data[slices] = values


_backend = Backend('ram', MemConnection, MemDataset, MemDataChunk)
