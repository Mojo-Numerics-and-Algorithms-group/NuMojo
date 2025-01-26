



# decompositions

##  Module Summary
  

## compute_householder


```Mojo
compute_householder[dtype: DType](mut H: Matrix[dtype], mut R: Matrix[dtype], row: Int, column: Int)
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- H
- R
- row
- column

## compute_qr


```Mojo
compute_qr[dtype: DType](mut H: Matrix[dtype], work_index: Int, mut A: Matrix[dtype], row_start: Int, column_start: Int)
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- H
- work_index
- A
- row_start
- column_start

## lu_decomposition


```Mojo
lu_decomposition[dtype: DType](A: NDArray[dtype]) -> Tuple[NDArray[dtype], NDArray[dtype]]
```  
Summary  
  
Perform LU (lower-upper) decomposition for array.  
  
Parameters:  

- dtype: Data type of the upper and upper triangular matrices.
  
Args:  

- A: Input matrix for decomposition. It should be a row-major matrix.


For efficiency, `dtype` of the output arrays will be the same as the input
array. Thus, use `astype()` before passing the array to this function.

Example:
```
import numojo as nm
fn main() raises:
    var arr = nm.NDArray[nm.f64]("[[1,2,3], [4,5,6], [7,8,9]]")
    var U: nm.NDArray
    var L: nm.NDArray
    L, U = nm.linalg.lu_decomposition(arr)
    print(arr)
    print(L)
    print(U)
```
```console
[[      1.0     2.0     3.0     ]
 [      4.0     5.0     6.0     ]
 [      7.0     8.0     9.0     ]]
2-D array  Shape: [3, 3]  DType: float64
[[      1.0     0.0     0.0     ]
 [      4.0     1.0     0.0     ]
 [      7.0     2.0     1.0     ]]
2-D array  Shape: [3, 3]  DType: float64
[[      1.0     2.0     3.0     ]
 [      0.0     -3.0    -6.0    ]
 [      0.0     0.0     0.0     ]]
2-D array  Shape: [3, 3]  DType: float64
```

Further readings:
- Linear Algebra And Its Applications, fourth edition, Gilbert Strang
- https://en.wikipedia.org/wiki/LU_decomposition
- https://www.scicoding.com/how-to-calculate-lu-decomposition-in-python/
- https://courses.physics.illinois.edu/cs357/sp2020/notes/ref-9-linsys.html.

```Mojo
lu_decomposition[dtype: DType](A: Matrix[dtype]) -> Tuple[Matrix[dtype], Matrix[dtype]]
```  
Summary  
  
Perform LU (lower-upper) decomposition for matrix.  
  
Parameters:  

- dtype
  
Args:  

- A

## partial_pivoting


```Mojo
partial_pivoting[dtype: DType](owned A: NDArray[dtype]) -> Tuple[NDArray[dtype], NDArray[dtype], Int]
```  
Summary  
  
Perform partial pivoting for a square matrix.  
  
Parameters:  

- dtype
  
Args:  

- A: 2-d square array.


```Mojo
partial_pivoting[dtype: DType](owned A: Matrix[dtype]) -> Tuple[Matrix[dtype], Matrix[dtype], Int]
```  
Summary  
  
Perform partial pivoting for matrix.  
  
Parameters:  

- dtype
  
Args:  

- A

## qr


```Mojo
qr[dtype: DType](owned A: Matrix[dtype]) -> Tuple[Matrix[dtype], Matrix[dtype]]
```  
Summary  
  
Compute the QR decomposition of a matrix.  
  
Parameters:  

- dtype
  
Args:  

- A: The input matrix to be factorized.


Decompose the matrix `A` as `QR`, where `Q` is orthonormal and `R` is upper-triangular.
This function is similar to `numpy.linalg.qr`.
