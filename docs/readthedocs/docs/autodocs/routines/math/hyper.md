



# hyper

##  Module Summary
  
Implements Hyperbolic functions for arrays.
## arccosh


```Mojo
arccosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## acosh


```Mojo
acosh[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Apply acosh also known as inverse hyperbolic cosine .  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array: An Array.


```Mojo
acosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## arcsinh


```Mojo
arcsinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## asinh


```Mojo
asinh[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Apply asinh also known as inverse hyperbolic sine .  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array: An Array.


```Mojo
asinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## arctanh


```Mojo
arctanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## atanh


```Mojo
atanh[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Apply atanh also known as inverse hyperbolic tangent .  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array: An Array.


```Mojo
atanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## cosh


```Mojo
cosh[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Apply cosh also known as hyperbolic cosine .  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array: An Array assumed to be in radian.


```Mojo
cosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## sinh


```Mojo
sinh[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Apply sin also known as hyperbolic sine .  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array: An Array assumed to be in radian.


```Mojo
sinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## tanh


```Mojo
tanh[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Apply tan also known as hyperbolic tangent .  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array: An Array assumed to be in radian.


```Mojo
tanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A
