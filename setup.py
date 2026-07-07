#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from setuptools import setup, Extension
from Cython.Build import build_ext

import numpy

_here = os.path.dirname(os.path.realpath(__file__))

setup(name='snowmatching',
      # Building the DTW C Core
      ext_modules=[
          Extension("snowmatching.DTW.DTW_CCore",
                    ["snowmatching/DTW/DTW_CCore.pyx"],
                    libraries=["m"],
                    extra_compile_args = ["-O3", "-march=native", "-fopenmp"],
                    include_dirs=[numpy.get_include()],
                    # extra_link_args=['-fopenmp']
                    )],
      cmdclass = {"build_ext": build_ext},
      )
