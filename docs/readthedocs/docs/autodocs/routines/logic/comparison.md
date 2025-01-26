



# comparison

##  Module Summary
  
Implements comparison math currently not using backend due to bool bitpacking issue
## greater


```Mojo
greater[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[bool]
```  
Summary  
  
Performs element-wise check of whether values in x are greater than values in y.  
  
Parameters:  

- dtype: The dtype of the input NDArray.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array1: First NDArray to compare.
- array2: Second NDArray to compare.


```Mojo
greater[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) -> NDArray[bool]
```  
Summary  
  
Performs element-wise check of whether values in x are greater than a scalar.  
  
Parameters:  

- dtype: The dtype of the input NDArray.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array1: First NDArray to compare.
- scalar: Scalar to compare.

## greater_equal


```Mojo
greater_equal[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[bool]
```  
Summary  
  
Performs element-wise check of whether values in x are greater than or equal to values in y.  
  
Parameters:  

- dtype: The dtype of the input NDArray.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array1: First NDArray to compare.
- array2: Second NDArray to compare.


```Mojo
greater_equal[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) -> NDArray[bool]
```  
Summary  
  
Performs element-wise check of whether values in x are greater than or equal to a scalar.  
  
Parameters:  

- dtype: The dtype of the input NDArray.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array1: First NDArray to compare.
- scalar: Scalar to compare.

## less


```Mojo
less[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[bool]
```  
Summary  
  
Performs element-wise check of whether values in x are to values in y.  
  
Parameters:  

- dtype: The dtype of the input NDArray.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array1: First NDArray to compare.
- array2: Second NDArray to compare.


```Mojo
less[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) -> NDArray[bool]
```  
Summary  
  
Performs element-wise check of whether values in x are to a scalar.  
  
Parameters:  

- dtype: The dtype of the input NDArray.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array1: First NDArray to compare.
- scalar: Scalar to compare.

## less_equal


```Mojo
less_equal[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[bool]
```  
Summary  
  
Performs element-wise check of whether values in x are less than or equal to values in y.  
  
Parameters:  

- dtype: The dtype of the input NDArray.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array1: First NDArray to compare.
- array2: Second NDArray to compare.


```Mojo
less_equal[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) -> NDArray[bool]
```  
Summary  
  
Performs element-wise check of whether values in x are less than or equal to a scalar.  
  
Parameters:  

- dtype: The dtype of the input NDArray.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array1: First NDArray to compare.
- scalar: Scalar to compare.

## equal


```Mojo
equal[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[bool]
```  
Summary  
  
Performs element-wise check of whether values in x are equal to values in y.  
  
Parameters:  

- dtype: The dtype of the input NDArray.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array1: First NDArray to compare.
- array2: Second NDArray to compare.


```Mojo
equal[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) -> NDArray[bool]
```  
Summary  
  
Performs element-wise check of whether values in x are equal to a scalar.  
  
Parameters:  

- dtype: The dtype of the input NDArray.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array1: First NDArray to compare.
- scalar: Scalar to compare.

## not_equal


```Mojo
not_equal[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[bool]
```  
Summary  
  
Performs element-wise check of whether values in x are not equal to values in y.  
  
Parameters:  

- dtype: The dtype of the input NDArray.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array1: First NDArray to compare.
- array2: Second NDArray to compare.


```Mojo
not_equal[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) -> NDArray[bool]
```  
Summary  
  
Performs element-wise check of whether values in x are not equal to values in y.  
  
Parameters:  

- dtype: The dtype of the input NDArray.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array1: First NDArray to compare.
- scalar: Scalar to compare.
