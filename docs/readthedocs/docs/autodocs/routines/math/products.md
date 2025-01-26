



# products

##  Module Summary
  

## prod


```Mojo
prod[dtype: DType](A: NDArray[dtype]) -> SIMD[dtype, 1]
```  
Summary  
  
Returns products of all items in the array.  
  
Parameters:  

- dtype
  
Args:  

- A: NDArray.


Example:
```console
> print(A)
[[      0.1315377950668335      0.458650141954422       0.21895918250083923     ]
[      0.67886471748352051     0.93469291925430298     0.51941639184951782     ]
[      0.034572109580039978    0.52970021963119507     0.007698186207562685    ]]
2-D array  Shape: [3, 3]  DType: float32

> print(nm.prod(A))
6.1377261317829834e-07
```


```Mojo
prod[dtype: DType](A: NDArray[dtype], owned axis: Int) -> NDArray[dtype]
```  
Summary  
  
Returns products of array elements over a given axis.  
  
Parameters:  

- dtype
  
Args:  

- A: NDArray.
- axis: The axis along which the product is performed.


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
from numojo import Matrix
var A = Matrix.rand(shape=(100, 100))
print(mat.prod(A, axis=0))
print(mat.prod(A, axis=1))
```
## cumprod


```Mojo
cumprod[dtype: DType](A: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Returns cumprod of all items of an array. The array is flattened before cumprod.  
  
Parameters:  

- dtype: The element type.
  
Args:  

- A: NDArray.


```Mojo
cumprod[dtype: DType](owned A: NDArray[dtype], owned axis: Int) -> NDArray[dtype]
```  
Summary  
  
Returns cumprod of array by axis.  
  
Parameters:  

- dtype: The element type.
  
Args:  

- A: NDArray.
- axis: Axis.


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
from numojo import Matrix
var A = Matrix.rand(shape=(100, 100))
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
from numojo import Matrix
var A = Matrix.rand(shape=(100, 100))
print(mat.cumprod(A, axis=0))
print(mat.cumprod(A, axis=1))
```