



# interpolate

##  Module Summary
  
Interpolate Module - Implements interpolation functions
## interp1d


```Mojo
interp1d[dtype: DType = float64](xi: NDArray[dtype], x: NDArray[dtype], y: NDArray[dtype], type: String = String("linear"), fill_method: String = String("interpolate")) -> NDArray[dtype]
```  
Summary  
  
Interpolate the values of y at the points xi.  
  
Parameters:  

- dtype: The element type. Defualt: `float64`
  
Args:  

- xi: An Array.
- x: An Array.
- y: An Array.
- type: The interpolation method. Default: String("linear")
- fill_method: The fill value. Default: String("interpolate")
