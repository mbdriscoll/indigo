import os
from distutils.core import setup, Extension
import numpy as np

exts = []

customcpu_backend = Extension('slo.backends._customcpu',
    sources = ['slo/backends/_customcpu.c'],
    include_dirs=[np.get_include()],
    extra_compile_args = ['-std=c11', '-fopenmp', '-m64', '-O3'],
    extra_link_args=['-fopenmp', '-mavx'],
)
exts.append(customcpu_backend)

if os.path.exists( 'slo/backends/libgpu.a' ):
    customgpu_backend = Extension('slo.backends._customgpu',
        sources = ['slo/backends/_customgpu.c'],
        include_dirs=[np.get_include()],
        extra_compile_args = ['-std=c11'],
        extra_link_args=['slo/backends/libgpu.a', '-L/usr/local/cuda/lib64', '-lcudart', '-lcublas', '-lstdc++'],
    )
    exts.append(customgpu_backend)

setup(
    name = 'slo',
    packages = ['slo', 'slo.backends'],
    version = '1.0.0',
    ext_modules = exts,
)
