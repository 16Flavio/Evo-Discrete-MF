import os
import sys
import pybind11
from setuptools import setup, Extension

"""
Setup script for building the C++ extension module 'fast_solver'.
Uses pybind11 for Python bindings and OpenMP for parallelization.
"""

# Determine the include directory for pybind11
pybind11_include = pybind11.get_include()

# Define the extension module
ext_modules = [
    Extension(
        "fast_solver",
        ["fast_solver.cpp"],
        include_dirs=[
            pybind11_include,
            "eigen-3.4.0", 
            os.path.join(sys.prefix, 'include'),
            os.path.join(sys.prefix, 'Library', 'include') 
        ],
        language="c++",
        extra_compile_args=["/O2", "/openmp"] if sys.platform == "win32" else ["-O3", "-fopenmp"],
        extra_link_args=[] if sys.platform == "win32" else ["-fopenmp"],
    ),
]

setup(
    name="fast_solver",
    version="2.0",
    author="Antigravity",
    description="Optimized C++ Solver for Matrix Factorization",
    ext_modules=ext_modules,
)