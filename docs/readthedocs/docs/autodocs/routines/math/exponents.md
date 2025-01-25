



# exponents

##  Module Summary
  

## Aliases
  
`ln`: Natural Log equivelent to log
## exp


```Mojo
exp[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Calculate element-wise euler's constant(e) to the power of NDArray[i].  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- array: A NDArray.

## exp2


```Mojo
exp2[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Calculate element-wise two to the power of NDArray[i].  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- array: A NDArray.

## expm1


```Mojo
expm1[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Calculate element-wise euler's constant(e) to the power of NDArray[i] minus1.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- array: A NDArray.

## log


```Mojo
log[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Element-wise natural logarithm of NDArray.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- array: A NDArray.

## log2


```Mojo
log2[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Element-wise logarithm base two of NDArray.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- array: A NDArray.

## log10


```Mojo
log10[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Element-wise logarithm base ten of NDArray.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- array: A NDArray.

## log1p


```Mojo
log1p[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Element-wise natural logarithm of 1 plus NDArray.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized`. Defualt: `Vectorized`
  
Args:  

- array: A NDArray.
