



# solving

##  Module Summary
  
Linear Algebra Solver
## forward_substitution


```Mojo
forward_substitution[dtype: DType](L: NDArray[dtype], y: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Perform forward substitution to solve `Lx = y`.  
  
Parameters:  

- dtype
  
Args:  

- L: A lower triangular matrix.
- y: A vector.


Paramters:
    dtype: dtype of the resulting vector.

## back_substitution


```Mojo
back_substitution[dtype: DType](U: NDArray[dtype], y: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Perform forward substitution to solve `Ux = y`.  
  
Parameters:  

- dtype
  
Args:  

- U: A upper triangular matrix.
- y: A vector.


Paramters:
    dtype: dtype of the resulting vector.

## inv


```Mojo
inv[dtype: DType](A: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Find the inverse of a non-singular, row-major matrix.  
  
Parameters:  

- dtype: Data type of the inverse matrix.
  
Args:  

- A: Input matrix. It should be non-singular, square, and row-major.


It uses the function `solve()` to solve `AB = I` for B, where I is
an identity matrix.

The speed is faster than numpy for matrices smaller than 100x100,
and is slower for larger matrices.


```Mojo
inv[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
Inverse of matrix.  
  
Parameters:  

- dtype
  
Args:  

- A

## inv_lu


```Mojo
inv_lu[dtype: DType](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Find the inverse of a non-singular, row-major matrix.  
  
Parameters:  

- dtype: Data type of the inverse matrix.
  
Args:  

- array: Input matrix. It should be non-singular, square, and row-major.


Use LU decomposition algorithm.

The speed is faster than numpy for matrices smaller than 100x100,
and is slower for larger matrices.

TODO: Fix the issues in parallelization.
`AX = I` where `I` is an identity matrix.

## lstsq


```Mojo
lstsq[dtype: DType](X: Matrix[dtype], y: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
Caclulate the OLS estimates.  
  
Parameters:  

- dtype
  
Args:  

- X
- y


Example:
```mojo
from numojo import Matrix
X = Matrix.rand((1000000, 5))
y = Matrix.rand((1000000, 1))
print(mat.lstsq(X, y))
```
```console
[[0.18731374756029967]
 [0.18821352688798607]
 [0.18717162200411439]
 [0.1867570378683612]
 [0.18828715376701158]]
Size: 5x1  DType: float64
```
## solve


```Mojo
solve[dtype: DType](A: NDArray[dtype], Y: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Solve the linear system `AX = Y` for `X`.  
  
Parameters:  

- dtype: Data type of the inversed matrix.
  
Args:  

- A: Non-singular, square, and row-major matrix. The size is m x m.
- Y: Matrix of size m x n.


`A` should be a non-singular, row-major matrix (m x m).
`Y` should be a matrix of (m x n).
`X` is a matrix of (m x n).
LU decomposition algorithm is adopted.

The speed is faster than numpy for matrices smaller than 100x100,
and is slower for larger matrices.

For efficiency, `dtype` of the output array will be the same as the input
arrays. Thus, use `astype()` before passing the arrays to this function.

TODO: Use LAPACK for large matrices when it is available.

An example goes as follows.

```mojo
import numojo as nm
fn main() raises:
    var A = nm.fromstring("[[1, 0, 1], [0, 2, 1], [1, 1, 1]]")
    var B = nm.fromstring("[[1, 0, 0], [0, 1, 0], [0, 0, 1]]")
    var X = nm.linalg.solve(A, B)
    print(X)
```
```console
[[      -1.0    -1.0    2.0     ]
 [      -1.0    0.0     1.0     ]
 [      2.0     1.0     -2.0    ]]
2-D array  Shape: [3, 3]  DType: float64
```

The example is also a way to calculate inverse of matrix.

```Mojo
solve[dtype: DType](A: Matrix[dtype], Y: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
Solve `AX = Y` using LUP decomposition.  
  
Parameters:  

- dtype
  
Args:  

- A
- Y

## solve_lu


```Mojo
solve_lu[dtype: DType](A: Matrix[dtype], Y: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
Solve `AX = Y` using LU decomposition.  
  
Parameters:  

- dtype
  
Args:  

- A
- Y
