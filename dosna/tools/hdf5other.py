import h5py
import h5json
import numpy as np

# NumPy arrays examples
a = np.arange(15)
b = np.array([7,8,9])
c = np.arange(10)

f = h5py.File("testlinks.h5", "w")

B = f.create_group("B")
Ba = f.create_group("Ba")
Bar = f.create_group("Bar")
Bazz = f.create_group("Bazz")
Battt = Bazz.create_group("Battt")
Bappp = Bazz.create_group("Bapppp")
Bafffff = Bazz.create_group("Bafffff")

print(Bazz.get("Bafffff"))
#print(f.values())

