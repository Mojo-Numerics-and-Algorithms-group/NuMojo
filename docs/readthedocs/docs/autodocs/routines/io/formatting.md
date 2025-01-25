



# formatting

##  Module Summary
  

## format_float_scientific


```Mojo
format_float_scientific[dtype: DType = float64](x: SIMD[dtype, 1], precision: Int = 10, sign: Bool = False) -> String
```  
Summary  
  
Format a float in scientific notation.  
  
Parameters:  

- dtype: Datatype of the float. Defualt: `float64`
  
Args:  

- x: The float to format.
- precision: The number of decimal places to include in the mantissa. Default: 10
- sign: Whether to include the sign of the float in the result. Defaults to False. Default: False
