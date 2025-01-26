



# products

##  Module Summary
  
Matrix and vector products
## cross


```Mojo
cross[dtype: DType = float64](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Compute the cross product of two arrays.  
  
Parameters:  

- dtype Defualt: `float64`
  
Constraints:

`array1` and `array2` must be of shape (3,).  
  
Args:  

- array1: A array.
- array2: A array.


Parameters
    dtype: The element type.

## dot


```Mojo
dot[dtype: DType = float64](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Compute the dot product of two arrays.  
  
Parameters:  

- dtype Defualt: `float64`
  
Constraints:

`array1` and `array2` must be 1 dimensional.  
  
Args:  

- array1: A array.
- array2: A array.


Parameters
    dtype: The element type.

## tile


```Mojo
tile[: origin.set, //, tiled_fn: fn[Int, Int](Int, Int) capturing -> None, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int)
```  
Summary  
  
  
  
Parameters:  

- 
- tiled_fn
- tile_x
- tile_y
  
Args:  

- end_x
- end_y

## matmul_tiled_unrolled_parallelized


```Mojo
matmul_tiled_unrolled_parallelized[dtype: DType](A: NDArray[dtype], B: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Matrix multiplication vectorized, tiled, unrolled, and parallelized.  
  
Parameters:  

- dtype
  
Args:  

- A
- B

## matmul_1darray


```Mojo
matmul_1darray[dtype: DType](A: NDArray[dtype], B: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Array multiplication for 1-d arrays (inner dot).  
  
Parameters:  

- dtype
  
Args:  

- A
- B

## matmul_2darray


```Mojo
matmul_2darray[dtype: DType](A: NDArray[dtype], B: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Array multiplication for 2-d arrays (inner dot).  
  
Parameters:  

- dtype
  
Args:  

- A: First array.
- B: Second array.


Parameter:
    dtype: Data type.

Return:
    A multiplied by B.

Notes:
    The multiplication is vectorized and parallelized.

References:
    [1] https://docs.modular.com/mojo/notebooks/Matmul.
    Compared to the reference, we increases the size of
    the SIMD vector from the default width to 16. The purpose is to
    increase the performance via SIMD.
    This reduces the execution time by ~50 percent compared to
    `matmul_parallelized` and `matmul_tiled_unrolled_parallelized` for large
    matrices.
## matmul


```Mojo
matmul[dtype: DType](A: NDArray[dtype], B: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Array multiplication for any dimensions.  
  
Parameters:  

- dtype
  
Args:  

- A: First array.
- B: Second array.


Parameter:
    dtype: Data type.

Return:
    A multiplied by B.

Notes:

    When A and B are 1darray, it is equal to dot of vectors:
    `(i) @ (i) -> (1)`.

    When A and B are 2darray, it is equal to inner products of matrices:
    `(i,j) @ (j,k) -> (i,k)`.

    When A and B are more than 2d, it is equal to a stack of 2darrays:
    `(i,j,k) @ (i,k,l) -> (i,j,l)` and
    `(i,j,k,l) @ (i,j,l,m) -> (i,j,k,m)`.

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


Example:
```mojo
from numojo import Matrix
var A = Matrix.rand(shape=(1000, 1000))
var B = Matrix.rand(shape=(1000, 1000))
var C = mat.matmul(A, B)
```
## matmul_naive


```Mojo
matmul_naive[dtype: DType](A: NDArray[dtype], B: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Matrix multiplication with three nested loops.  
  
Parameters:  

- dtype
  
Args:  

- A
- B
