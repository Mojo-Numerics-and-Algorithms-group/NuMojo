



# logic

##  Module Summary
  
`numojo.mat.logic` module provides logic functions for Matrix type.
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

## any


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
