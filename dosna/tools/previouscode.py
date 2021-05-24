
"""
RECURSING HDF5DICT AND STORING GROUPS IN THE DOSNADICTIONARY
"""
            for key, value in hdf5dict.items():
                if isinstance(value, LazyHdfDict):
                    dosnadict[key] = {}
                    g = dosnaobject.create_group(key)
                    print(g, g.name)
                    dosnadict[key] = _recurse(value, dosnadict[key])
                elif isinstance(value, h5py._hl.dataset.Dataset):
                    if value.nbytes > 10:
                        dataset = dosnaobject.create_dataset(
                            key,
                            shape=value.shape,
                            dtype=value.dtype,
                            chunk_size=value.chunks,
                        )
                        if value.chunks is not None:
                            for s in value.iter_chunks():
                                dataset[s] = value[s]
 
                        dosnadict[key] = dataset
                    else:
                        arr = np.zeros(value.shape)
                        value.read_direct(arr)
                        
            return dosnadict
        
        return _recurse(hdf5dict, {})


"""
PATH TO DICT
"""
def path_to_dict(path_name, datadict):
    r = re.compile("(.*/)")  # this checks for anything and only two backlash
    if r.match(path_name) is not None:
        elements = path_name.split("/")
        for key in elements:
            if key not in datadict:
                print(
                    "Key: {} not found".format(key)
                )  # maybe full length #check what msg h5py gives #raise error
                return None
            else:
                datadict = datadict[key]
    return datadict


#geog = {"Spain": {"City": {"Madrid": {"Barrio"}}}}
#path_name = "Spain/City/Madrid"
#dict_path = path_to_dict(path_name, geog)

"""
DOSNA DATASET CHUNKS TEST

con = dn.Connection("dosna-test")
con.connect()
ds = con.create_dataset("data", data=np.random.randn(10, 10, 10), chunk_size=(5, 5, 5))
np_data = np.full(shape=(1, 5, 5, 5), fill_value=[1, 2, 3, 4, 5])
ds_slices = list(ds.data_chunks.keys())
for i in ds_slices:
    chunk = ds.get_chunk(i)
    data = chunk.get_data()  # slices dentro del chunk get the initial position
    chunk.set_data(np_data)
"""


"""
CREATING TREE FROM HDF5DICT

fname = "testfile1.h5"
res_lazy = hd.load(fname, lazy=True)

    def create_tree(self, name, connection_handler):
        log.debug("Creating Tree `%s`", name)
        tree = MemTree(self, name)
        def _recurse(treeobject):
            for key, value in treeobject.items():
                if key == self.name:
                    treeobject[self.name][treename] = {}
                    for k, v in value.items():
                        if k == treename:
                            value[treename] = {}
                            value[treename][treename] = tree
                else:
                    if isinstance(value, dict):
                        _recurse(value)
            return treeobject

        datadict = _recurse(connection.trees)
        return tree
"""

"""
TREE METHODS

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
            print("BackendTree not found") 
        log.debug("Removing Backend `%s`", name)
        del self.trees[name]



class MemTree(): 

    
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

        Creates a new empty group and gives it a name
        :param location: identifier of the file/group in a file with respect to which the new group is to be identified
        :param path: string that provides wither an absolute path or a relative path to the new group
                     Begins with a slash: absolute path indicating that it locates the new group from the root group of the HDF5 file.
                     No slash: relative path is a path from that file's root group.
                               when the location is a group, a relative path is a path from that group.

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
        
        
    def link(self, n1, n2, e):
        global graph
        if n1 not in graph:
            raise Exception("Node", n1, "does not exist")
        if n2 not in graph:
            raise Exception("Node", n2, "does not exist")
        else:
            temp = [n2, e] 
            graph[n1].append(e) 
        return e



