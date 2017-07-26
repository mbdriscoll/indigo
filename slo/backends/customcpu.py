from slo.backends.mkl import MklBackend
from slo.backends._customcpu import csrmm

class CustomCpuBackend(MklBackend):
    def ccsrmm(self, y, A_shape, A_indx, A_ptr, A_vals, x, alpha, beta, adjoint=False):
        pass
