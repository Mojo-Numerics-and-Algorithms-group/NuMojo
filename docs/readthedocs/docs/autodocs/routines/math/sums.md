



# sums

##  Module Summary
  

## sum


```Mojo
sum[dtype: DType](A: NDArray[dtype]) -> SIMD[dtype, 1]
```  
Summary  
  
Returns sum of all items in the array.  
  
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
> print(nm.sum(A))
3.5140917301177979
```


```Mojo
sum[dtype: DType](A: NDArray[dtype], owned axis: Int) -> NDArray[dtype]
```  
Summary  
  
Returns sums of array elements over a given axis.  
  
Parameters:  

- dtype
  
Args:  

- A: NDArray.
- axis: The axis along which the sum is performed.


Example:
```mojo
import numojo as nm
var A = nm.random.randn(100, 100)
print(nm.sum(A, axis=0))
```


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
from numojo import Matrix
var A = Matrix.rand(shape=(100, 100))
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
from numojo import Matrix
var A = Matrix.rand(shape=(100, 100))
print(mat.sum(A, axis=0))
print(mat.sum(A, axis=1))
```
## cumsum


```Mojo
cumsum[dtype: DType](A: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Returns cumsum of all items of an array. The array is flattened before cumsum.  
  
Parameters:  

- dtype: The element type.
  
Args:  

- A: NDArray.


```Mojo
cumsum[dtype: DType](owned A: NDArray[dtype], owned axis: Int) -> NDArray[dtype]
```  
Summary  
  
Returns cumsum of array by axis.  
  
Parameters:  

- dtype: The element type.
  
Args:  

- A: NDArray.
- axis: Axis.


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
from numojo import Matrix
var A = Matrix.rand(shape=(100, 100))
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
from numojo import Matrix
var A = Matrix.rand(shape=(100, 100))
print(mat.cumsum(A, axis=0))
print(mat.cumsum(A, axis=1))
```