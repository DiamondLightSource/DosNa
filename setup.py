#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='DosNa',
    version='0.1',
    description='Distributed Object-Store Numpy Array (DosNa)',
    author='Imanol Luengo',
    author_email='imanol.luengo@diamond.ac.uk',
    url='https://github.com/DiamondLightSource/DosNa',
    include_package_data=True,
    packages=find_packages() + ['dosna']
)
