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

    return backends

def get_backend(name):
    if name == 'mkl':
        from slo.backends.mkl  import MklBackend
        return MklBackend
    elif name == 'cuda':
        from slo.backends.cuda import CudaBackend
        return CudaBackend
    elif name == 'numpy':
        from slo.backends.np   import NumpyBackend
        return NumpyBackend
    elif name == 'customcpu':
        from slo.backends.customcpu import CustomCpuBackend
        return CustomCpuBackend
    else:
        log.error("unrecognized backend: %s", name)
