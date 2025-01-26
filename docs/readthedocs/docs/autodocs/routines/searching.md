



# searching

##  Module Summary
  

## argmax


```Mojo
argmax[dtype: DType](array: NDArray[dtype]) -> Int
```  
Summary  
  
Argmax of a array.  
  
Parameters:  

- dtype: The element type.
  
Args:  

- array: A array.


```Mojo
argmax[dtype: DType](A: Matrix[dtype]) -> SIMD[index, 1]
```  
Summary  
  
Index of the max. It is first flattened before sorting.  
  
Parameters:  

- dtype
  
Args:  

- A


```Mojo
argmax[dtype: DType](A: Matrix[dtype], axis: Int) -> Matrix[index]
```  
Summary  
  
Index of the max along the given axis.  
  
Parameters:  

- dtype
  
Args:  

- A
- axis

## argmin


```Mojo
argmin[dtype: DType](array: NDArray[dtype]) -> Int
```  
Summary  
  
Argmin of a array. Parameters:     dtype: The element type.  
  
Parameters:  

- dtype
  
Args:  

- array: A array.


```Mojo
argmin[dtype: DType](A: Matrix[dtype]) -> SIMD[index, 1]
```  
Summary  
  
Index of the min. It is first flattened before sorting.  
  
Parameters:  

- dtype
  
Args:  

- A


```Mojo
argmin[dtype: DType](A: Matrix[dtype], axis: Int) -> Matrix[index]
```  
Summary  
  
Index of the min along the given axis.  
  
Parameters:  

- dtype
  
Args:  

- A
- axis
