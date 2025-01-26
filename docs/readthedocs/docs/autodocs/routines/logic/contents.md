



# contents

##  Module Summary
  
Implements Checking routines: currently not SIMD due to bool bit packing issue
## isinf


```Mojo
isinf[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[bool]
```  
Summary  
  
Checks if each element of the input array is infinite.  
  
Parameters:  

- dtype: DType - Data type of the input array.
- backend: _mf.Backend - Backend to use for the operation. Defaults to _mf.Vectorized. Defualt: `Vectorized`
  
Args:  

- array: NDArray[dtype] - Input array to check.

## isfinite


```Mojo
isfinite[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[bool]
```  
Summary  
  
Checks if each element of the input array is finite.  
  
Parameters:  

- dtype: DType - Data type of the input array.
- backend: _mf.Backend - Backend to use for the operation. Defaults to _mf.Vectorized. Defualt: `Vectorized`
  
Args:  

- array: NDArray[dtype] - Input array to check.

## isnan


```Mojo
isnan[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[bool]
```  
Summary  
  
Checks if each element of the input array is NaN.  
  
Parameters:  

- dtype: DType - Data type of the input array.
- backend: _mf.Backend - Backend to use for the operation. Defaults to _mf.Vectorized. Defualt: `Vectorized`
  
Args:  

- array: NDArray[dtype] - Input array to check.
