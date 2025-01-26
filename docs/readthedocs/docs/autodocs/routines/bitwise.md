



# bitwise

##  Module Summary
  

## invert


```Mojo
invert[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Element-wise invert of an array.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Constraints:

The array must be either a boolean or integral array.  
  
Args:  

- array: A NDArray.
