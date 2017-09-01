import os
from distutils.core import setup, Extension
import numpy as np

exts = []

customcpu_backend = Extension('indigo.backends._customcpu',
    sources = ['indigo/backends/_customcpu.c'],
    include_dirs=[np.get_include()],
    extra_compile_args = ['-std=c11', '-fopenmp', '-m64', '-O3', '-Wno-unknown-pragmas'],
    extra_link_args=['-fopenmp', '-mavx'],
)
exts.append(customcpu_backend)

if os.path.exists( 'indigo/backends/libgpu.a' ):
    customgpu_backend = Extension('indigo.backends._customgpu',
        sources = ['indigo/backends/_customgpu.c'],
        include_dirs=[np.get_include()],
        extra_compile_args = ['-std=c11'],
        extra_link_args=['indigo/backends/libgpu.a', '-L/usr/local/cuda/lib64', '-lcudart', '-lcublas', '-lstdc++'],
    )
    exts.append(customgpu_backend)

setup(
    name = 'indigo',
    packages = ['indigo', 'indigo.backends'],
    version = '1.0.0',
    author = 'Michael Driscoll',
    author_email = 'mbdriscoll@gmail.com',
    license = 'BSD',
    url = 'https://mbdriscoll.github.io/indigo/',
    download_url = 'https://github.com/mbdriscoll/indigo/archive/master.zip',
    ext_modules = exts,
)
