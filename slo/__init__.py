import logging

log = logging.getLogger(__name__)

_the_backend = None

def init(backend='numpy', device_id=0):
    from slo.backends.backend import Backend

    if backend == 'numpy':
        from slo.backends.numpy import NumpyBackend
        BackendCls = NumpyBackend
    elif backend == 'mkl':
        from slo.backends.mkl import MklBackend
        BackendCls = MklBackend
    elif backend == 'cuda':
        from slo.backends.cuda import CudaBackend
        BackendCls = CudaBackend
    elif issubclass(backend, Backend):
        BackendCls = backend
    else:
        raise ValueError("no such backend: %s" % backend)

    global _the_backend
    _the_backend = BackendCls(device_id)


def backend():
    return _the_backend
