



# mathematics

##  Module Summary
  
`numojo.mat.mathematics` module provides mathematical functions for Matrix type.
## sin


```Mojo
sin[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## cos


```Mojo
cos[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## tan


```Mojo
tan[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## arcsin


```Mojo
arcsin[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## asin


```Mojo
asin[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## arccos


```Mojo
arccos[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## acos


```Mojo
acos[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## arctan


```Mojo
arctan[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## atan


```Mojo
atan[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## sinh


```Mojo
sinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## cosh


```Mojo
cosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## tanh


```Mojo
tanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## arcsinh


```Mojo
arcsinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## asinh


```Mojo
asinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## arccosh


```Mojo
arccosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## acosh


```Mojo
acosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## arctanh


```Mojo
arctanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## atanh


```Mojo
atanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## round


```Mojo
round[dtype: DType](owned A: Matrix[dtype], decimals: Int = 0) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A
- decimals Default: 0

## sum


```Mojo
sum[dtype: DType](A: Matrix[dtype]) -> SIMD[dtype, 1]
```  
Summary  
  
Sum up all items in the Matrix.  
  
Parameters:  

- dtype
  
Args:  

- A: Matrix.


Example:
```mojo
from numojo import mat
var A = mat.rand(shape=(100, 100))
print(mat.sum(A))
```

```Mojo
sum[dtype: DType](A: Matrix[dtype], axis: Int) -> Matrix[dtype]
```  
Summary  
  
Sum up the items in a Matrix along the axis.  
  
Parameters:  

- dtype
  
Args:  

- A: Matrix.
- axis: 0 or 1.


Example:
```mojo
from numojo import mat
var A = mat.rand(shape=(100, 100))
print(mat.sum(A, axis=0))
print(mat.sum(A, axis=1))
```
## prod


```Mojo
prod[dtype: DType](A: Matrix[dtype]) -> SIMD[dtype, 1]
```  
Summary  
  
Product of all items in the Matrix.  
  
Parameters:  

- dtype
  
Args:  

- A: Matrix.


```Mojo
prod[dtype: DType](A: Matrix[dtype], axis: Int) -> Matrix[dtype]
```  
Summary  
  
Product of items in a Matrix along the axis.  
  
Parameters:  

- dtype
  
Args:  

- A: Matrix.
- axis: 0 or 1.


Example:
```mojo
from numojo import mat
var A = mat.rand(shape=(100, 100))
print(mat.prod(A, axis=0))
print(mat.prod(A, axis=1))
```
## cumsum


```Mojo
cumsum[dtype: DType](owned A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
Cumsum of flattened matrix.  
  
Parameters:  

- dtype
  
Args:  

- A: Matrix.


Example:
```mojo
from numojo import mat
var A = mat.rand(shape=(100, 100))
print(mat.cumsum(A))
```

```Mojo
cumsum[dtype: DType](owned A: Matrix[dtype], axis: Int) -> Matrix[dtype]
```  
Summary  
  
Cumsum of Matrix along the axis.  
  
Parameters:  

- dtype
  
Args:  

- A: Matrix.
- axis: 0 or 1.


Example:
```mojo
from numojo import mat
var A = mat.rand(shape=(100, 100))
print(mat.cumsum(A, axis=0))
print(mat.cumsum(A, axis=1))
```
## cumprod


```Mojo
cumprod[dtype: DType](owned A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
Cumprod of flattened matrix.  
  
Parameters:  

- dtype
  
Args:  

- A: Matrix.


Example:
```mojo
from numojo import mat
var A = mat.rand(shape=(100, 100))
print(mat.cumprod(A))
```

```Mojo
cumprod[dtype: DType](owned A: Matrix[dtype], axis: Int) -> Matrix[dtype]
```  
Summary  
  
Cumprod of Matrix along the axis.  
  
Parameters:  

- dtype
  
Args:  

- A: Matrix.
- axis: 0 or 1.


Example:
```mojo
from numojo import mat
var A = mat.rand(shape=(100, 100))
print(mat.cumprod(A, axis=0))
print(mat.cumprod(A, axis=1))
```