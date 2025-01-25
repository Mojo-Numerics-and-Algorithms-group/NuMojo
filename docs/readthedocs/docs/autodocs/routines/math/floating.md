



# floating

##  Module Summary
  

## copysign


```Mojo
copysign[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Copy the sign of the first NDArray and apply it to the second NDArray.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Constraints:

Both arrays must have the same shapes.  
  
Args:  

- array1: A NDArray.
- array2: A NDArray.
