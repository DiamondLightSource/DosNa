from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(cmdclass = {'build_ext': build_ext},
      ext_modules = [ Extension('pyclovis', ['pyclovis.pyx'],
                                include_dirs=['.'],
                                libraries=['clovis',
                                           'mero'],
                                library_dirs=['.'],
                                extra_compile_args=['-ggdb',
                                                    '-Wall',
                                                    '-Werror']),])
