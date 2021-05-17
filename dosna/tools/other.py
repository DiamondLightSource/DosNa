import re
import dosna as dn
import numpy as np
import hdf5todict.hdf5todict as hd

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


geog = {"Spain": {"City": {"Madrid": {"Barrio"}}}}
path_name = "Spain/City/Madrid"
dict_path = path_to_dict(path_name, geog)

"""

con = dn.Connection("dosna-test")
con.connect()
ds = con.create_dataset("data", data=np.random.randn(10, 10, 10), chunk_size=(5, 5, 5))
np_data = np.full(shape=(1, 5, 5, 5), fill_value=[1, 2, 3, 4, 5])
ds_slices = list(ds.data_chunks.keys())
for i in ds_slices:
    chunk = ds.get_chunk(i)
    data = chunk.get_data()  # slices dentro del chunk get the initial position
    chunk.set_data(np_data)


fname = "testfile1.h5"
res_lazy = hd.load(fname, lazy=True)
print(res_lazy)

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

cn = dn.Connection("dn-ram")
A = cn.create_node("location", "/")
B = A.create_node("B")
Ba = A.create_node("Ba")
Bar = A.create_node("Bar")
Bazz = A.create_node("Bazz")
Battt = Bazz.create_node("Battt")
Bappp = Bazz.create_node("Bapppp")
Bafffff = Bazz.create_node("Bafffff")
Dataset1 = Battt.create_node("Dataset1")
#print(A.iterate())
print(Bazz.iterate())
#print(A.get_node(""))
#print(Bazz.links.keys())
#print(A.iterate())
#A.visit()
#A.get_object_info()
print(A.get_node("Bazz/Battt/Dataset1"))
#print(Bazz.__contains__("Battfssft"))
#print(Bazz.keys())
#print(Bazz.values())
#print(Bazz.items())
#C = A.create_node("location", "C")
#D = B.create_node("location", "D")
#A.unlink("A", "C", "t")
#v = A.iterate()
#print(v)

#Ideal result
#<HDF5 group "/Bazz" (3 members)>