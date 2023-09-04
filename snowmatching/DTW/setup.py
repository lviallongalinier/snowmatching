'''
Created on 24 aout 2016

@author: hagenmullerp
'''

from setuptools import setup, Extension
from Cython.Build import build_ext

import numpy
# remove openmp if non parallel
ext_modules = [
    Extension("DTW_CCore",
              ["DTW_CCore.pyx"],
              libraries=["m"],
              # Be careful -ffast-math disables nan checks, including isnan function
              extra_compile_args = ["-O3", "-march=native", "-fopenmp"],
              include_dirs=[numpy.get_include()],
              # extra_link_args=['-fopenmp']  # To remove if non parallel
              )
]

setup(name = "DTW_CCore",
      cmdclass = {"build_ext": build_ext},
      ext_modules = ext_modules
      )
