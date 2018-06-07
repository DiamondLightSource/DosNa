#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='DosNa',
    version='0.1',
    description='Distributed Object-Store Numpy Array (DosNa)',
    author='Imanol Luengo',
    author_email='imanol.luengo@diamond.ac.uk',
    url='https://github.com/DiamondLightSource/DosNa',
    package_data={
        'dosna.support.pyclovis': ['*.pxd', '*.pxi', '*.pyx', '*.c', '*.h',
                                   '*_config', '*.conf', '*.so']
    },
    packages=find_packages() + ['dosna'],
    python_requires=">=2.7.0",
    install_requires=["numpy>=1.13.0", "six>=1.10.0"],
    extras_require={
        "hdf5": ["h5py>=2.7.0"],
        "jl": ["joblib>=0.11"],
        "ceph": ["python-cephlibs>=0.94.0"],
        "mpi": ["mpi4py>=3.0.0"]
    }
)
