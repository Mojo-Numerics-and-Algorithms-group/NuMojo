



# linalg

##  Module Summary
  
`numojo.mat.linalg` module provides functions for linear algebra.
## matmul


```Mojo
matmul[dtype: DType](A: Matrix[dtype], B: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
Matrix multiplication.  
  
Parameters:  

- dtype
  
Args:  

- A
- B


See `numojo.math.linalg.matmul.matmul_parallelized()`.

Example:
```mojo
from numojo import mat
var A = mat.rand(shape=(1000, 1000))
var B = mat.rand(shape=(1000, 1000))
var C = mat.matmul(A, B)
```
## partial_pivoting


```Mojo
partial_pivoting[dtype: DType](owned A: Matrix[dtype]) -> Tuple[Matrix[dtype], Matrix[dtype], Int]
```  
Summary  
  
Perform partial pivoting.  
  
Parameters:  

- dtype
  
Args:  

- A

## lu_decomposition


```Mojo
lu_decomposition[dtype: DType](A: Matrix[dtype]) -> Tuple[Matrix[dtype], Matrix[dtype]]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## det


```Mojo
det[dtype: DType](A: Matrix[dtype]) -> SIMD[dtype, 1]
```  
Summary  
  
Find the determinant of A using LUP decomposition.  
  
Parameters:  

- dtype
  
Args:  

- A

## trace


```Mojo
trace[dtype: DType](A: Matrix[dtype], offset: Int = 0) -> SIMD[dtype, 1]
```  
Summary  
  
Return the sum along diagonals of the array.  
  
Parameters:  

- dtype
  
Args:  

- A
- offset Default: 0


Similar to `numpy.trace`.
## inv


```Mojo
inv[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
Inverse of matrix.  
  
Parameters:  

- dtype
  
Args:  

- A

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
from numojo import mat
X = mat.rand((1000000, 5))
y = mat.rand((1000000, 1))
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

## transpose


```Mojo
transpose[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
Transpose of matrix.  
  
Parameters:  

- dtype
  
Args:  

- A
