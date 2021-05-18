import re
import dosna as dn
import numpy as np
import hdf5todict.hdf5todict as hd

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

























