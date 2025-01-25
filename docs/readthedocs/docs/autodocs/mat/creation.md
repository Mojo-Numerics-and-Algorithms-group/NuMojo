



# creation

##  Module Summary
  
`numojo.mat.creation` module provides functions for creating matrix.
## full


```Mojo
full[dtype: DType = float64](shape: Tuple[Int, Int], fill_value: SIMD[dtype, 1] = SIMD(0)) -> Matrix[dtype]
```  
Summary  
  
Return a matrix with given shape and filled value.  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- shape
- fill_value Default: SIMD(0)


Example:
```mojo
from numojo import mat
var A = mat.full(shape=(10, 10), fill_value=100)
```
## zeros


```Mojo
zeros[dtype: DType = float64](shape: Tuple[Int, Int]) -> Matrix[dtype]
```  
Summary  
  
Return a matrix with given shape and filled with zeros.  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- shape


Example:
```mojo
from numojo import mat
var A = mat.zeros(shape=(10, 10))
```
## ones


```Mojo
ones[dtype: DType = float64](shape: Tuple[Int, Int]) -> Matrix[dtype]
```  
Summary  
  
Return a matrix with given shape and filled with ones.  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- shape


Example:
```mojo
from numojo import mat
var A = mat.ones(shape=(10, 10))
```
## identity


```Mojo
identity[dtype: DType = float64](len: Int) -> Matrix[dtype]
```  
Summary  
  
Return a matrix with given shape and filled value.  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- len


Example:
```mojo
from numojo import mat
var A = mat.identity(12)
```
## rand


```Mojo
rand[dtype: DType = float64](shape: Tuple[Int, Int]) -> Matrix[dtype]
```  
Summary  
  
Return a matrix with random values uniformed distributed between 0 and 1.  
  
Parameters:  

- dtype: The data type of the NDArray elements. Defualt: `float64`
  
Args:  

- shape: The shape of the Matrix.


Example:
```mojo
from numojo import mat
var A = mat.rand((12, 12))
```

## fromlist


```Mojo
fromlist[dtype: DType](object: List[SIMD[dtype, 1]], shape: Tuple[Int, Int] = Tuple(VariadicPack(<store_to_mem({0}), store_to_mem({0})>, True))) -> Matrix[dtype]
```  
Summary  
  
Create a matrix from a 1-dimensional list into given shape.  
  
Parameters:  

- dtype
  
Args:  

- object
- shape Default: Tuple(VariadicPack(<store_to_mem({0}), store_to_mem({0})>, True))


If no shape is passed, the return matrix will be a row vector.

Example:
```mojo
from numojo import mat
fn main() raises:
    print(mat.fromlist(List[Float64](1, 2, 3, 4, 5), (5, 1)))
```
## fromstring


```Mojo
fromstring[dtype: DType = float64](text: String, shape: Tuple[Int, Int] = Tuple(VariadicPack(<store_to_mem({0}), store_to_mem({0})>, True))) -> Matrix[dtype]
```  
Summary  
  
Matrix initialization from string representation of an matrix.  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- text: String representation of a matrix.
- shape: Shape of the matrix. Default: Tuple(VariadicPack(<store_to_mem({0}), store_to_mem({0})>, True))


Comma, right brackets, and whitespace are treated as seperators of numbers.
Digits, underscores, and minus signs are treated as a part of the numbers.

If now shape is passed, the return matrix will be a row vector.

Example:
```mojo
from numojo.prelude import *
from numojo import mat
fn main() raises:
    var A = mat.fromstring[f32](
    "1 2 .3 4 5 6.5 7 1_323.12 9 10, 11.12, 12 13 14 15 16", (4, 4))
```
```console
[[1.0   2.0     0.30000001192092896     4.0]
 [5.0   6.5     7.0     1323.1199951171875]
 [9.0   10.0    11.119999885559082      12.0]
 [13.0  14.0    15.0    16.0]]
Size: 4x4  DType: float32
```
