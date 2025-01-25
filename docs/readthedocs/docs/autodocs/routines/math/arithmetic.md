



# arithmetic

##  Module Summary
  
Implements arithmetic operations functions
## add


```Mojo
add[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Perform addition on two arrays.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Constraints:

Both arrays must have the same shapes.  
  
Args:  

- array1: A NDArray.
- array2: A NDArray.


```Mojo
add[dtype: DType, backend: Backend = Vectorized, *, OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](array1: NDArray[dtype], array2: NDArray[OtherDType]) -> NDArray[ResultDType]
```  
Summary  
  
Perform addition on two arrays.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defualts to `Vectorized`. Defualt: `Vectorized`
- OtherDType: The element type of the second array.
- ResultDType: The element type of the result array. Defualt: `result[::DType,::DType]()`
  
Constraints:

Both arrays must have the same shapes.  
  
Args:  

- array1: A NDArray.
- array2: A NDArray.


```Mojo
add[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype], scalar: SIMD[dtype, 1]) -> NDArray[dtype]
```  
Summary  
  
Perform addition on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- array: A NDArray.
- scalar: A NDArray.


```Mojo
add[dtype: DType, backend: Backend = Vectorized, *, OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](array: NDArray[dtype], scalar: SIMD[OtherDType, 1]) -> NDArray[ResultDType]
```  
Summary  
  
Perform addition on two arrays.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defualts to `Vectorized`. Defualt: `Vectorized`
- OtherDType: The element type of the second array.
- ResultDType: The element type of the result array. Defualt: `result[::DType,::DType]()`
  
Constraints:

Both arrays must have the same shapes.  
  
Args:  

- array: A NDArray.
- scalar: A NDArray.


```Mojo
add[dtype: DType, backend: Backend = Vectorized](scalar: SIMD[dtype, 1], array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Perform addition on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- scalar: A NDArray.
- array: A NDArray.


```Mojo
add[dtype: DType, backend: Backend = Vectorized, *, OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](scalar: SIMD[dtype, 1], array: NDArray[OtherDType]) -> NDArray[ResultDType]
```  
Summary  
  
Perform addition on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defualts to `Vectorized`. Defualt: `Vectorized`
- OtherDType: The element type of the second array.
- ResultDType: The element type of the result array. Defualt: `result[::DType,::DType]()`
  
Args:  

- scalar: A NDArray.
- array: A NDArray.


```Mojo
add[dtype: DType, backend: Backend = Vectorized](owned *values: Variant[NDArray[dtype], SIMD[dtype, 1]]) -> NDArray[dtype]
```  
Summary  
  
Perform addition on a list of arrays and a scalars.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- \*values: A list of arrays or Scalars to be added.

## sub


```Mojo
sub[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Perform subtraction on two arrays.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Constraints:

Both arrays must have the same shapes.  
  
Args:  

- array1: A NDArray.
- array2: A NDArray.


```Mojo
sub[dtype: DType, backend: Backend = Vectorized, *, OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](array1: NDArray[dtype], array2: NDArray[OtherDType]) -> NDArray[ResultDType]
```  
Summary  
  
Perform subtraction on two arrays.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defualts to `Vectorized`. Defualt: `Vectorized`
- OtherDType: The element type of the second array.
- ResultDType: The element type of the result array. Defualt: `result[::DType,::DType]()`
  
Constraints:

Both arrays must have the same shapes.  
  
Args:  

- array1: A NDArray.
- array2: A NDArray.


```Mojo
sub[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype], scalar: SIMD[dtype, 1]) -> NDArray[dtype]
```  
Summary  
  
Perform subtraction on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- array: A NDArray.
- scalar: A NDArray.


```Mojo
sub[dtype: DType, backend: Backend = Vectorized, *, OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](array: NDArray[dtype], scalar: SIMD[OtherDType, 1]) -> NDArray[ResultDType]
```  
Summary  
  
Perform subtraction on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defualts to `Vectorized`. Defualt: `Vectorized`
- OtherDType: The element type of the second array.
- ResultDType: The element type of the result array. Defualt: `result[::DType,::DType]()`
  
Args:  

- array: A NDArray.
- scalar: A NDArray.


```Mojo
sub[dtype: DType, backend: Backend = Vectorized](scalar: SIMD[dtype, 1], array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Perform subtraction on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- scalar: A NDArray.
- array: A NDArray.


```Mojo
sub[dtype: DType, backend: Backend = Vectorized, *, OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](scalar: SIMD[dtype, 1], array: NDArray[OtherDType]) -> NDArray[ResultDType]
```  
Summary  
  
Perform subtraction on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defualts to `Vectorized`. Defualt: `Vectorized`
- OtherDType: The element type of the second array.
- ResultDType: The element type of the result array. Defualt: `result[::DType,::DType]()`
  
Args:  

- scalar: A NDArray.
- array: A NDArray.

## diff


```Mojo
diff[dtype: DType = float64](array: NDArray[dtype], n: Int) -> NDArray[dtype]
```  
Summary  
  
Compute the n-th order difference of the input array.  
  
Parameters:  

- dtype: The element type. Defualt: `float64`
  
Args:  

- array: A array.
- n: The order of the difference.

## mod


```Mojo
mod[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Element-wise modulo of array1 and array2.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Constraints:

Both arrays must have the same shapes.  
  
Args:  

- array1: A NDArray.
- array2: A NDArray.


```Mojo
mod[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype], scalar: SIMD[dtype, 1]) -> NDArray[dtype]
```  
Summary  
  
Perform subtraction on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- array: A NDArray.
- scalar: A NDArray.


```Mojo
mod[dtype: DType, backend: Backend = Vectorized](scalar: SIMD[dtype, 1], array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Perform subtraction on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- scalar: A NDArray.
- array: A NDArray.

## mul


```Mojo
mul[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Element-wise product of array1 and array2.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Constraints:

Both arrays must have the same shapes.  
  
Args:  

- array1: A NDArray.
- array2: A NDArray.


```Mojo
mul[dtype: DType, backend: Backend = Vectorized, *, OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](array1: NDArray[dtype], array2: NDArray[OtherDType]) -> NDArray[ResultDType]
```  
Summary  
  
Perform multiplication on between two arrays.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defualts to `Vectorized`. Defualt: `Vectorized`
- OtherDType: The element type of the second array.
- ResultDType: The element type of the result array. Defualt: `result[::DType,::DType]()`
  
Args:  

- array1: A NDArray.
- array2: A NDArray.


```Mojo
mul[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype], scalar: SIMD[dtype, 1]) -> NDArray[dtype]
```  
Summary  
  
Perform multiplication on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- array: A NDArray.
- scalar: A NDArray.


```Mojo
mul[dtype: DType, backend: Backend = Vectorized, *, OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](array: NDArray[dtype], scalar: SIMD[OtherDType, 1]) -> NDArray[ResultDType]
```  
Summary  
  
Perform multiplication on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defualts to `Vectorized`. Defualt: `Vectorized`
- OtherDType: The element type of the second array.
- ResultDType: The element type of the result array. Defualt: `result[::DType,::DType]()`
  
Args:  

- array: A NDArray.
- scalar: A NDArray.


```Mojo
mul[dtype: DType, backend: Backend = Vectorized](scalar: SIMD[dtype, 1], array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Perform multiplication on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- scalar: A NDArray.
- array: A NDArray.


```Mojo
mul[dtype: DType, backend: Backend = Vectorized, *, OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](scalar: SIMD[dtype, 1], array: NDArray[OtherDType]) -> NDArray[ResultDType]
```  
Summary  
  
Perform multiplication on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defualts to `Vectorized`. Defualt: `Vectorized`
- OtherDType: The element type of the second array.
- ResultDType: The element type of the result array. Defualt: `result[::DType,::DType]()`
  
Args:  

- scalar: A NDArray.
- array: A NDArray.


```Mojo
mul[dtype: DType, backend: Backend = Vectorized](owned *values: Variant[NDArray[dtype], SIMD[dtype, 1]]) -> NDArray[dtype]
```  
Summary  
  
Perform multiplication on a list of arrays an arrays and a scalars.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- \*values: A list of arrays or Scalars to be added.

## div


```Mojo
div[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Element-wise quotent of array1 and array2.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Constraints:

Both arrays must have the same shapes.  
  
Args:  

- array1: A NDArray.
- array2: A NDArray.


```Mojo
div[dtype: DType, backend: Backend = Vectorized, *, OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](array1: NDArray[dtype], array2: NDArray[OtherDType]) -> NDArray[ResultDType]
```  
Summary  
  
Perform true division on between two arrays.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defualts to `Vectorized`. Defualt: `Vectorized`
- OtherDType: The element type of the second array.
- ResultDType: The element type of the result array. Defualt: `result[::DType,::DType]()`
  
Args:  

- array1: A NDArray.
- array2: A NDArray.


```Mojo
div[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype], scalar: SIMD[dtype, 1]) -> NDArray[dtype]
```  
Summary  
  
Perform true division on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- array: A NDArray.
- scalar: A NDArray.


```Mojo
div[dtype: DType, backend: Backend = Vectorized, *, OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](array: NDArray[dtype], scalar: SIMD[OtherDType, 1]) -> NDArray[ResultDType]
```  
Summary  
  
Perform true division on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defualts to `Vectorized`. Defualt: `Vectorized`
- OtherDType: The element type of the second array.
- ResultDType: The element type of the result array. Defualt: `result[::DType,::DType]()`
  
Args:  

- array: A NDArray.
- scalar: A NDArray.


```Mojo
div[dtype: DType, backend: Backend = Vectorized](scalar: SIMD[dtype, 1], array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Perform true division on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- scalar: A NDArray.
- array: A NDArray.


```Mojo
div[dtype: DType, backend: Backend = Vectorized, *, OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](scalar: SIMD[dtype, 1], array: NDArray[OtherDType]) -> NDArray[ResultDType]
```  
Summary  
  
Perform true division on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defualts to `Vectorized`. Defualt: `Vectorized`
- OtherDType: The element type of the second array.
- ResultDType: The element type of the result array. Defualt: `result[::DType,::DType]()`
  
Args:  

- scalar: A NDArray.
- array: A NDArray.

## floor_div


```Mojo
floor_div[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Element-wise quotent of array1 and array2.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Constraints:

Both arrays must have the same shapes.  
  
Args:  

- array1: A NDArray.
- array2: A NDArray.


```Mojo
floor_div[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype], scalar: SIMD[dtype, 1]) -> NDArray[dtype]
```  
Summary  
  
Perform true division on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- array: A NDArray.
- scalar: A NDArray.


```Mojo
floor_div[dtype: DType, backend: Backend = Vectorized](scalar: SIMD[dtype, 1], array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Perform true division on between an array and a scalar.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- scalar: A NDArray.
- array: A NDArray.

## fma


```Mojo
fma[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype], array3: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Apply a SIMD level fuse multipy add function of three variables and one return to a NDArray.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Constraints:

Both arrays must have the same shape.  
  
Args:  

- array1: A NDArray.
- array2: A NDArray.
- array3: A NDArray.


```Mojo
fma[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype], simd: SIMD[dtype, 1]) -> NDArray[dtype]
```  
Summary  
  
Apply a SIMD level fuse multipy add function of three variables and one return to a NDArray.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Constraints:

Both arrays must have the same shape  
  
Args:  

- array1: A NDArray.
- array2: A NDArray.
- simd: A SIMD[dtype,1] value to be added.

## remainder


```Mojo
remainder[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Element-wise remainders of NDArray.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Constraints:

Both arrays must have the same shapes.  
  
Args:  

- array1: A NDArray.
- array2: A NDArray.
