



# differences

##  Module Summary
  

## gradient


```Mojo
gradient[dtype: DType = float64](x: NDArray[dtype], spacing: SIMD[dtype, 1]) -> NDArray[dtype]
```  
Summary  
  
Compute the gradient of y over x using the trapezoidal rule.  
  
Parameters:  

- dtype: Input data type. Defualt: `float64`
  
Constraints:

`fdtype` must be a floating-point type if `idtype` is not a floating-point type.  
  
Args:  

- x: An array.
- spacing: An array of the same shape as x containing the spacing between adjacent elements.

## trapz


```Mojo
trapz[dtype: DType = float64](y: NDArray[dtype], x: NDArray[dtype]) -> SIMD[dtype, 1]
```  
Summary  
  
Compute the integral of y over x using the trapezoidal rule.  
  
Parameters:  

- dtype: The element type. Defualt: `float64`
  
Constraints:

`x` and `y` must have the same shape. `fdtype` must be a floating-point type if `idtype` is not a floating-point type.  
  
Args:  

- y: An array.
- x: An array.
