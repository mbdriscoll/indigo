Indigo implements a collection of routines for constructing and evaluating high-performance linear operators on multicore and accelerator platforms.

## Installation

### From Python Package Index
```
pip install indigo
```

### From source

1. Install Anaconda or Miniconda.
2. Download `indigo`.
```
git clone git@github.com:mbdriscoll/indigo.git
cd indigo
```
3. Create a `conda` environment and install `indigo`'s dependences.
```
conda env create --file requirements.txt
```
With gcc
```
conda develop -b .
```
With icc
```
LDSHARED='icc -shared' CC=icc conda develop -b .
```
For the optional customgpu backend (requires nvcc)
```
make -C indigo/backends
```

4. (Optional) Run the test suite.
```
pytest
```

## Examples
See [spmm.py](https://github.com/mbdriscoll/indigo/blob/master/examples/spmm.py) in the `examples` directory.

## Available Backends

Indigo provides three backends. Users need not change any application code to move between them.
* Numpy: reference backend. Slow, but widely available and provides informative error messages.
* MKL: multicore CPU backend utilizing Intel's FFT and Sparse Blas libraries.
* CUDA: GPU backend utilizing nVidia's CUFFT and CUSPARSE libraries.

To select a backend, import and instantiate it:
```python
from indigo.backends import get_backend
b = get_backend('cuda', device_id=1)
```

## Available Operators
Indigo provides a number of classes for constructing structured linear operators.

### Simple Operators
Simple operators implement linear transformations directly via high-performance libraries.

#### Diagonal Matrix (`indigo.operators.Diag`)
![Diagonal Matrix Image](imgs/DiagM.png)

#### Sparse Matrix (`indigo.operators.SpMatrix`)
![Sparse Matrix Image](imgs/SparseM.png)

#### Dense Matrix (`indigo.operators.DenseMatrix`)
![Dense Matrix Image](imgs/DenseM.png)

#### DFT Matrix (`indigo.operators.UnscaledFFT`)
![DFT Matrix Image](imgs/FFT.png)
Supports 3D FFTs. 1D and 2D TBD.


### Composite Operators

Composite operators represent a collection of operators arranged in some structured way.


#### BlockDiag Matrix (`indigo.operators.BlockDiag`)
![Block Diagonal Matrix](imgs/BlockDiag.png)

Represents different submatrices arranged along a diagonal.

#### KronI Matrix (`indigo.operators.KronI`)

![KronI Matrix](imgs/KronI.png)

The `KronI` operator represents repetition of the same submatrix along a diagonal.


#### VStack Matrix (`indigo.operators.VStack`)

![VStack Matrix](imgs/VStack.png)

Represents different submatrices stacked vertically. Akin to `scipy.sparse.vstack`.

#### HStack Matrix (`indigo.operators.HStack`)

Currently unimplemented. Consider using `VStack` and `Adjoint`:
```
HStack(A,B) = Adjoint( VStack( Adjoint(A), Adjoint(B) ) )
```

#### Product Matrix (`indigo.operators.Product`)

![Product Matrix](imgs/Product.png)


#### Adjoint Matrix (`indigo.operators.Adjoint`)

![Adjoint Matrix](imgs/Adjoint.png)


### Derived Operators
We can combine the aforementioned operators to implement higher-level functionality.

#### Unitary DFT matrix (`indigo.operators.UnitaryFFT`)

![UFFT Matrix](imgs/UnitaryFFT.png)

The scaling effect of the DFT can be undone by an elementwise multiplication, represented in Indigo as a diagonal matrix.

#### Centered DFT matrix (`indigo.operators.CenteredFFT`)

![CFFT Matrix](imgs/CenteredFFT.png)

A centered DFT consists of an FFT Shift, followed by a standard FFT, followed by another FFT Shift.


#### Non-uniform Fourier Transform (`indigo.operators.NUFFT`)

![NUFFT Matrix](imgs/NUFFT.png)

Indigo implements an NUFFT as a product of diagonal, FFT, and general sparse matrices (for apodization, FFT, and interpolation, respectively).

## FAQ
1. *What datatypes are supported?* `indigo` only support single-precision complex floating point numbers at the moment, but it's not a fundamental limitation.
