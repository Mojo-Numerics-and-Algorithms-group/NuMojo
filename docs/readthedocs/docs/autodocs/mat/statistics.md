



# statistics

##  Module Summary
  
`numojo.mat.statistics` module provides statistical functions for Matrix type.
## mean


```Mojo
mean[dtype: DType](A: Matrix[dtype]) -> SIMD[dtype, 1]
```  
Summary  
  
Calculate the arithmetic average of all items in the Matrix.  
  
Parameters:  

- dtype
  
Args:  

- A: Matrix.


```Mojo
mean[dtype: DType](A: Matrix[dtype], axis: Int) -> Matrix[dtype]
```  
Summary  
  
Calculate the arithmetic average of a Matrix along the axis.  
  
Parameters:  

- dtype
  
Args:  

- A: Matrix.
- axis: 0 or 1.

## variance


```Mojo
variance[dtype: DType](A: Matrix[dtype], ddof: Int = 0) -> SIMD[dtype, 1]
```  
Summary  
  
Compute the variance.  
  
Parameters:  

- dtype
  
Args:  

- A: Matrix.
- ddof: Delta degree of freedom. Default: 0


```Mojo
variance[dtype: DType](A: Matrix[dtype], axis: Int, ddof: Int = 0) -> Matrix[dtype]
```  
Summary  
  
Compute the variance along axis.  
  
Parameters:  

- dtype
  
Args:  

- A: Matrix.
- axis: 0 or 1.
- ddof: Delta degree of freedom. Default: 0

## std


```Mojo
std[dtype: DType](A: Matrix[dtype], ddof: Int = 0) -> SIMD[dtype, 1]
```  
Summary  
  
Compute the standard deviation.  
  
Parameters:  

- dtype
  
Args:  

- A: Matrix.
- ddof: Delta degree of freedom. Default: 0


```Mojo
std[dtype: DType](A: Matrix[dtype], axis: Int, ddof: Int = 0) -> Matrix[dtype]
```  
Summary  
  
Compute the standard deviation along axis.  
  
Parameters:  

- dtype
  
Args:  

- A: Matrix.
- axis: 0 or 1.
- ddof: Delta degree of freedom. Default: 0
