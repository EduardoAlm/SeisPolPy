from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np



setup(name="loopstft", ext_modules = cythonize(["loopstft.pyx"]), include_dirs=[np.get_include(),])