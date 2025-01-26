



# trig

##  Module Summary
  
Implements Trigonometry functions for arrays.
## arccos


```Mojo
arccos[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## acos


```Mojo
acos[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Apply acos also known as inverse cosine .  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array: An Array.


```Mojo
acos[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## arcsin


```Mojo
arcsin[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## asin


```Mojo
asin[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Apply asin also known as inverse sine .  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array: An Array.


```Mojo
asin[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## arctan


```Mojo
arctan[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## atan


```Mojo
atan[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Apply atan also known as inverse tangent .  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array: An Array.


```Mojo
atan[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## atan2


```Mojo
atan2[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Apply atan2 also known as inverse tangent. [atan2 wikipedia](https://en.wikipedia.org/wiki/Atan2).  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Constraints:

Both arrays must have the same shapes.  
  
Args:  

- array1: An Array.
- array2: An Array.

## cos


```Mojo
cos[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Apply cos also known as cosine.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array: An Array assumed to be in radian.


```Mojo
cos[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## sin


```Mojo
sin[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Apply sin also known as sine .  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array: An Array assumed to be in radian.


```Mojo
sin[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## tan


```Mojo
tan[dtype: DType, backend: Backend = Vectorized](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Apply tan also known as tangent .  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Args:  

- array: An Array assumed to be in radian.


```Mojo
tan[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype
  
Args:  

- A

## hypot


```Mojo
hypot[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Apply hypot also known as hypotenuse which finds the longest section of a right triangle given the other two sides.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Constraints:

Both arrays must have the same shapes.  
  
Args:  

- array1: An Array.
- array2: An Array.

## hypot_fma


```Mojo
hypot_fma[dtype: DType, backend: Backend = Vectorized](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Apply hypot also known as hypotenuse which finds the longest section of a right triangle given the other two sides.  
  
Parameters:  

- dtype: The element type.
- backend: Sets utility function origin, defaults to `Vectorized. Defualt: `Vectorized`
  
Constraints:

Both arrays must have the same shapes.  
  
Args:  

- array1: An Array.
- array2: An Array.
