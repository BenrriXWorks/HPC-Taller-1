from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

# Define la extensión optimizada
ext_modules = [
    Pybind11Extension(
        "ska_sort_py",
        ["src/ska_sort_binding.cpp"],
        include_dirs=[
            # Incluir el directorio de pybind11
            pybind11.get_include(),
            # Incluir el directorio de ska_sort
            "lib/",
        ],
        language='c++',
        cxx_std=17,  # ska_sort requiere C++17
        # Optimizaciones adicionales
        define_macros=[('VERSION_INFO', '"dev"')],
        # Flags de optimización agresiva
        extra_compile_args=['-O3', '-ffast-math'],
        extra_link_args=['-O3'],
    ),
    Pybind11Extension(
        "vergesort_py",
        ["src/vergesort_binding.cpp"],
        include_dirs=[
            # Incluir el directorio de pybind11
            pybind11.get_include(),
            # Incluir el directorio de vergesort y ska_sort
            "lib/",
            "lib/vergesort/",
        ],
        language='c++',
        cxx_std=17,  # vergesort requiere C++17
        # Optimizaciones adicionales
        define_macros=[('VERSION_INFO', '"dev"')],
        # Flags de optimización agresiva
        extra_compile_args=['-O3', '-ffast-math'],
        extra_link_args=['-O3'],
    ),
]

setup(
    name="ska_sort_py",
    version="2.5.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Python bindings for ska_sort and vergesort - Ultra-fast sorting algorithms",
    long_description="High-performance sorting algorithms: ska_sort (radix sort) and vergesort (adaptive hybrid sort with ska_sort fallback)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
