



# utility

##  Module Summary
  
Implements N-DIMENSIONAL ARRAY UTILITY FUNCTIONS
## fill_pointer


```Mojo
fill_pointer[dtype: DType](mut array: UnsafePointer[SIMD[dtype, 1]], size: Int, value: SIMD[dtype, 1])
```  
Summary  
  
Fill a NDArray with a specific value.  
  
Parameters:  

- dtype: The data type of the NDArray elements.
  
Args:  

- array: The pointer to the NDArray.
- size: The size of the NDArray.
- value: The value to fill the NDArray with.

## bool_to_numeric


```Mojo
bool_to_numeric[dtype: DType](array: NDArray[bool]) -> NDArray[dtype]
```  
Summary  
  
Convert a boolean NDArray to a numeric NDArray.  
  
Parameters:  

- dtype: The data type of the output NDArray elements.
  
Args:  

- array: The boolean NDArray to convert.

## to_numpy


```Mojo
to_numpy[dtype: DType](array: NDArray[dtype]) -> PythonObject
```  
Summary  
  
Convert a NDArray to a numpy array.  
  
Parameters:  

- dtype: The data type of the NDArray elements.
  
Args:  

- array: The NDArray to convert.


Example:
```console
var arr = NDArray[DType.float32](3, 3, 3)
var np_arr = to_numpy(arr)
var np_arr1 = arr.to_numpy()
```

## is_inttype


```Mojo
is_inttype[dtype: DType]() -> Bool
```  
Summary  
  
Check if the given dtype is an integer type at compile time.  
  
Parameters:  

- dtype: DType.


```Mojo
is_inttype(dtype: DType) -> Bool
```  
Summary  
  
Check if the given dtype is an integer type at run time.  
  
Args:  

- dtype: DType.

## is_floattype


```Mojo
is_floattype[dtype: DType]() -> Bool
```  
Summary  
  
Check if the given dtype is a floating point type at compile time.  
  
Parameters:  

- dtype: DType.


```Mojo
is_floattype(dtype: DType) -> Bool
```  
Summary  
  
Check if the given dtype is a floating point type at run time.  
  
Args:  

- dtype: DType.

## is_booltype


```Mojo
is_booltype[dtype: DType]() -> Bool
```  
Summary  
  
Check if the given dtype is a boolean type at compile time.  
  
Parameters:  

- dtype: DType.


```Mojo
is_booltype(dtype: DType) -> Bool
```  
Summary  
  
Check if the given dtype is a boolean type at run time.  
  
Args:  

- dtype: DType.
