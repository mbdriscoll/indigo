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

    return backends
