



# formatting

##  Module Summary
  

## Aliases
  
`DEFAULT_PRECISION`:   
`DEFAULT_SUPPRESS_SMALL`:   
`DEFAULT_SEPARATOR`:   
`DEFAULT_PADDING`:   
`DEFAULT_EDGE_ITEMS`:   
`DEFAULT_THRESHOLD`:   
`DEFAULT_LINE_WIDTH`:   
`DEFAULT_SIGN`:   
`DEFAULT_FLOAT_FORMAT`:   
`DEFAULT_COMPLEX_FORMAT`:   
`DEFAULT_NAN_STRING`:   
`DEFAULT_INF_STRING`:   
`DEFAULT_FORMATTED_WIDTH`:   
`DEFAULT_EXPONENT_THRESHOLD`:   
`DEFAULT_SUPPRESS_SCIENTIFIC`:   
`printoptions`: 
## PrintOptions

### PrintOptions Summary
  
  
  

### Parent Traits
  

- AnyType
- Copyable
- Movable
- UnknownDestructibility

### Fields
  
  
* precision `Int`  
* suppress_small `Bool`  
* separator `String`  
* padding `String`  
* threshold `Int`  
* line_width `Int`  
* edge_items `Int`  
* sign `Bool`  
* float_format `String`  
* complex_format `String`  
* nan_string `String`  
* inf_string `String`  
* formatted_width `Int`  
* exponent_threshold `Int`  
* suppress_scientific `Bool`  

### Functions

#### __init__


```Mojo
__init__(out self, precision: Int = 4, suppress_small: Bool = False, separator: String = String(" "), padding: String = String(""), threshold: Int = 10, line_width: Int = 75, edge_items: Int = 3, sign: Bool = False, float_format: String = String("fixed"), complex_format: String = String("parentheses"), nan_string: String = String("nan"), inf_string: String = String("inf"), formatted_width: Int = 8, exponent_threshold: Int = 4, suppress_scientific: Bool = False)
```  
Summary  
  
  
  
Args:  

- self
- precision Default: 4
- suppress_small Default: False
- separator Default: String(" ")
- padding Default: String("")
- threshold Default: 10
- line_width Default: 75
- edge_items Default: 3
- sign Default: False
- float_format Default: String("fixed")
- complex_format Default: String("parentheses")
- nan_string Default: String("nan")
- inf_string Default: String("inf")
- formatted_width Default: 8
- exponent_threshold Default: 4
- suppress_scientific Default: False

#### set_options


```Mojo
set_options(mut self, precision: Int = 4, suppress_small: Bool = False, separator: String = String(" "), padding: String = String(""), threshold: Int = 10, line_width: Int = 75, edge_items: Int = 3, sign: Bool = False, float_format: String = String("fixed"), complex_format: String = String("parentheses"), nan_string: String = String("nan"), inf_string: String = String("inf"), formatted_width: Int = 8, exponent_threshold: Int = 4, suppress_scientific: Bool = False)
```  
Summary  
  
  
  
Args:  

- self
- precision Default: 4
- suppress_small Default: False
- separator Default: String(" ")
- padding Default: String("")
- threshold Default: 10
- line_width Default: 75
- edge_items Default: 3
- sign Default: False
- float_format Default: String("fixed")
- complex_format Default: String("parentheses")
- nan_string Default: String("nan")
- inf_string Default: String("inf")
- formatted_width Default: 8
- exponent_threshold Default: 4
- suppress_scientific Default: False

#### __enter__


```Mojo
__enter__(mut self) -> Self
```  
Summary  
  
  
  
Args:  

- self

#### __exit__


```Mojo
__exit__(mut self)
```  
Summary  
  
  
  
Args:  

- self

## set_printoptions


```Mojo
set_printoptions(precision: Int = 4, suppress_small: Bool = False, separator: String = String(" "), padding: String = String(""), edge_items: Int = 3)
```  
Summary  
  
  
  
Args:  

- precision Default: 4
- suppress_small Default: False
- separator Default: String(" ")
- padding Default: String("")
- edge_items Default: 3

## format_floating_scientific


```Mojo
format_floating_scientific[dtype: DType = float64](x: SIMD[dtype, 1], precision: Int = 10, sign: Bool = False) -> String
```  
Summary  
  
Format a float in scientific notation.  
  
Parameters:  

- dtype: Datatype of the float. Defualt: `float64`
  
Args:  

- x: The float to format.
- precision: The number of decimal places to include in the mantissa. Default: 10
- sign: Whether to include the sign of the float in the result. Defaults to False. Default: False

## format_floating_precision


```Mojo
format_floating_precision[dtype: DType](value: SIMD[dtype, 1], precision: Int, sign: Bool = False) -> String
```  
Summary  
  
Format a floating-point value to the specified precision.  
  
Parameters:  

- dtype
  
Args:  

- value: The value to format.
- precision: The number of decimal places to include.
- sign: Whether to include the sign of the float in the result. Defaults to False. Default: False


```Mojo
format_floating_precision[cdtype: CDType, dtype: DType](value: ComplexSIMD[cdtype, dtype=dtype]) -> String
```  
Summary  
  
Format a complex floating-point value to the specified precision.  
  
Parameters:  

- cdtype
- dtype
  
Args:  

- value: The complex value to format.

## format_value


```Mojo
format_value[dtype: DType](value: SIMD[dtype, 1], print_options: PrintOptions) -> String
```  
Summary  
  
Format a single value based on the print options.  
  
Parameters:  

- dtype
  
Args:  

- value: The value to format.
- print_options: The print options.


```Mojo
format_value[cdtype: CDType, dtype: DType](value: ComplexSIMD[cdtype, dtype=dtype], print_options: PrintOptions) -> String
```  
Summary  
  
Format a complex value based on the print options.  
  
Parameters:  

- cdtype
- dtype
  
Args:  

- value: The complex value to format.
- print_options: The print options.
