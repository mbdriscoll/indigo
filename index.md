SLO implements a collection of routines for constructing and evaluating hig-performance linear operators on multicore CPU and GPU platforms.


## Available Operators
SLO provides a number of classes for constructing structured linear operators.


### Simple Operators
Simple operators implement linear transformations directly via high-performance libraries.

#### Sparse Matrix (`slo.operators.SpMatrix`)
![Sparse Matrix Image](imgs/SparseM.png)

#### Diagonal Matrix (`slo.operators.Diag`)
![Diagonal Matrix Image](imgs/DiagM.png)

#### DFT Matrix (`slo.operators.UnscaledFFT`)
![DFT Matrix Image](imgs/FFT.png)
Supports 3D FFTs. 1D and 2D TBD.


### Composite Operators
Composite operators represent a collection of other operators arranged in some structured way.

#### BlockDiag Matrix (`slo.operators.BlockDiag`)
![Block Diagonal Matrix](imgs/BlockDiag.png)

#### KronI Matrix (`slo.operators.KronI`)
![KronI Matrix](imgs/KronI.png)
The `KronI` operator represents repetition of a matrix along a diagonal.

#### VStack Matrix (`slo.operators.VStack`)
![VStack Matrix](imgs/VStack.png)

#### Product Matrix (`slo.operators.Product`)
TODO

#### Adjoint Matrix (`slo.operators.Adjoint`)
TODO


### Derived Operators
We can combine the aforementioned operators to implement higher-level functionality.

#### Unitary DFT matrix (`slo.operators.UnitaryFFT`)
![UFFT Matrix](imgs/UnitaryFFT.png)
The scaling effect of the DFT can be undone by an elementwise multiplication, represented in SLO as a diagonal matrix.

#### Centered DFT matrix (`slo.operators.CenteredFFT`)
![CFFT Matrix](imgs/CenteredFFT.png)
A centered DFT consists of an FFT Shift, followed by a standard FFT, followed by another FFT Shift.

#### Non-uniform Fourier Transform (`slo.operators.NUFFT`)
![NUFFT Matrix](imgs/NUFFT.png)
SLO implements an NUFFT as a product of diagonal, FFT, and general sparse matrices (for apodization, FFT, and interpolation, respectively).
