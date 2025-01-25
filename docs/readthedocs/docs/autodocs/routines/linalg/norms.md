



# norms

##  Module Summary
  

## det


```Mojo
det[dtype: DType](A: NDArray[dtype]) -> SIMD[dtype, 1]
```  
Summary  
  
Find the determinant of A using LUP decomposition.  
  
Parameters:  

- dtype
  
Args:  

- A


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
trace[dtype: DType](array: NDArray[dtype], offset: Int = 0, axis1: Int = 0, axis2: Int = 1) -> NDArray[dtype]
```  
Summary  
  
Computes the trace of a ndarray.  
  
Parameters:  

- dtype: Data type of the array.
  
Args:  

- array: A NDArray.
- offset: Offset of the diagonal from the main diagonal. Default: 0
- axis1: First axis. Default: 0
- axis2: Second axis. Default: 1


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