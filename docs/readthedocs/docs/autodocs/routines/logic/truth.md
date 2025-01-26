



# truth

##  Module Summary
  

## all


```Mojo
all[dtype: DType](A: Matrix[dtype]) -> SIMD[dtype, 1]
```  
Summary  
  
Test whether all array elements evaluate to True.  
  
Parameters:  

- dtype
  
Args:  

- A: Matrix.


```Mojo
all[dtype: DType](A: Matrix[dtype], axis: Int) -> Matrix[dtype]
```  
Summary  
  
Test whether all array elements evaluate to True along axis.  
  
Parameters:  

- dtype
  
Args:  

- A
- axis

## allt


```Mojo
allt(array: NDArray[bool]) -> SIMD[bool, 1]
```  
Summary  
  
If all True.  
  
Args:  

- array: A NDArray.

## any


```Mojo
any(array: NDArray[bool]) -> SIMD[bool, 1]
```  
Summary  
  
If any True.  
  
Args:  

- array: A NDArray.


```Mojo
any[dtype: DType](A: Matrix[dtype]) -> SIMD[dtype, 1]
```  
Summary  
  
Test whether any array elements evaluate to True.  
  
Parameters:  

- dtype
  
Args:  

- A: Matrix.


```Mojo
any[dtype: DType](A: Matrix[dtype], axis: Int) -> Matrix[dtype]
```  
Summary  
  
Test whether any array elements evaluate to True along axis.  
  
Parameters:  

- dtype
  
Args:  

- A
- axis
