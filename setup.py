#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

HERE = os.path.dirname(os.path.realpath(__file__))

PYTHON_MIN_VERSION = (3, 6)

packages = ['snowmatching', 'snowmatching/DTW']

setup(
        name='Snowmatching',
        version='0.1',
        description="Snowmatching is a collection of tools to match snow profiles.",
        long_description="""
        Snowmatching is a collection of tools to match snow profiles against a refernce
        profile for further use in comparisons of snow profiles from different sources,
        do clustering or define.
        """,

        # Building the DTW C Core
        ext_modules=[
            Extension("snowmatching.DTW.DTW_CCore",
                      ["snowmatching/DTW/DTW_CCore.pyx"],
                      libraries=["m"],
                      # Be careful -ffast-math disables nan checks, including isnan function
                      extra_compile_args = ["-O3", "-march=native", "-fopenmp"],
                      include_dirs=[numpy.get_include()],
                      # extra_link_args=['-fopenmp']
                     )
           ],
        cmdclass = {"build_ext": build_ext},

        # Python part of the code
        packages=packages,
        classifiers=[
            'Operating System :: POSIX :: Linux',

            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',

            'Topic :: Scientific/Engineering',
            ],
        install_requires=[
             'numpy',
             'scipy',
            ],
        python_requires='>=' + '.'.join(str(n) for n in PYTHON_MIN_VERSION),
        )
