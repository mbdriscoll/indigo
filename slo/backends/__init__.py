import logging

log = logging.getLogger(__name__)

def available_backends():
    backends = []

    try:
        from slo.backends.np import NumpyBackend
        backends.append( NumpyBackend )
    except Exception as e:
        log.warn("couldn't find NUMPY backend")

    try:
        from slo.backends.mkl import MklBackend
        backends.append( MklBackend )
    except Exception as e:
        log.warn("couldn't find MKL backend")

    try:
        from slo.backends.cuda import CudaBackend
        backends.append( CudaBackend )
    except Exception as e:
        log.warn("couldn't find CUDA backend")

    try:
        from slo.backends.customcpu import CustomCpuBackend
        backends.append( CustomCpuBackend )
    except Exception as e:
        log.warn("couldn't find CustomCpu backend")

    try:
        from slo.backends.customgpu import CustomGpuBackend
        backends.append( CustomGpuBackend )
    except Exception as e:
        log.warn("couldn't find CustomGpu backend")

    return backends

def get_backend(name, **init):
    """
    Instantiates the requested backend.
    """
    if name == 'mkl':
        from slo.backends.mkl  import MklBackend
        return MklBackend(**init)
    elif name == 'cuda':
        from slo.backends.cuda import CudaBackend
        return CudaBackend(**init)
    elif name == 'numpy':
        from slo.backends.np   import NumpyBackend
        return NumpyBackend(**init)
    elif name == 'customcpu':
        from slo.backends.customcpu import CustomCpuBackend
        return CustomCpuBackend(**init)
    elif name == 'customgpu':
        from slo.backends.customgpu import CustomGpuBackend
        return CustomGpuBackend(**init)
    else:
        log.error("unrecognized backend: %s", name)
