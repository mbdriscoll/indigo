from distutils.core import setup, Extension
import numpy as np

customcpu_backend = Extension('slo.backends._customcpu',
    sources = ['slo/backends/_customcpu.c'],
    include_dirs=[np.get_include()],
    extra_compile_args = ['-std=c11', '-fopenmp', '-m64', '-O3'],
    extra_link_args=['-fopenmp', '-mavx'],
)

setup(
    name = 'slo',
    packages = ['slo', 'slo.backends'],
    version = '1.0.0',
    ext_modules = [customcpu_backend],
)
