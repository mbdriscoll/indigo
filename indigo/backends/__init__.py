import os
import logging

log = logging.getLogger(__name__)

def available_backends():
    backends = []

    allow = os.environ.get("INDIGO_TEST_BACKENDS",
        "np,mkl,cuda,customcpu,customgpu")

    try:
        from indigo.backends.np import NumpyBackend
        if 'np' in allow: backends.append( NumpyBackend )
    except Exception as e:
        log.warn("couldn't find NUMPY backend")

    try:
        from indigo.backends.mkl import MklBackend
        if 'mkl' in allow: backends.append( MklBackend )
    except Exception as e:
        log.warn("couldn't find MKL backend")

    try:
        from indigo.backends.cuda import CudaBackend
        if 'cuda' in allow: backends.append( CudaBackend )
    except Exception as e:
        log.warn("couldn't find CUDA backend")

    try:
        from indigo.backends.customcpu import CustomCpuBackend
        if 'customcpu' in allow: backends.append( CustomCpuBackend )
    except Exception as e:
        log.warn("couldn't find CustomCpu backend")

    try:
        from indigo.backends.customgpu import CustomGpuBackend
        if 'customgpu' in allow: backends.append( CustomGpuBackend )
    except Exception as e:
        log.warn("couldn't find CustomGpu backend")

    return backends

def get_backend(name, **init):
    """
    Instantiates the requested backend.
    """
    if name == 'mkl':
        from indigo.backends.mkl  import MklBackend
        return MklBackend(**init)
    elif name == 'cuda':
        from indigo.backends.cuda import CudaBackend
        return CudaBackend(**init)
    elif name == 'numpy':
        from indigo.backends.np   import NumpyBackend
        return NumpyBackend(**init)
    elif name == 'customcpu':
        from indigo.backends.customcpu import CustomCpuBackend
        return CustomCpuBackend(**init)
    elif name == 'customgpu':
        from indigo.backends.customgpu import CustomGpuBackend
        return CustomGpuBackend(**init)
    else:
        raise ValueError("unrecognized backend: %s" % name)
