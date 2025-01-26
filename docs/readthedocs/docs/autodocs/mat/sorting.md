



# sorting

##  Module Summary
  
`numojo.mat.sorting` module provides sorting functions for Matrix type.
## argsort


```Mojo
argsort[dtype: DType](A: Matrix[dtype]) -> Matrix[index]
```  
Summary  
  
Argsort the Matrix. It is first flattened before sorting.  
  
Parameters:  

- dtype
  
Args:  

- A


```Mojo
argsort[dtype: DType](owned A: Matrix[dtype], axis: Int) -> Matrix[index]
```  
Summary  
  
Argsort the Matrix along the given axis.  
  
Parameters:  

- dtype
  
Args:  

- A
- axis

## sort


```Mojo
sort[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
Sort the Matrix. It is first flattened before sorting.  
  
Parameters:  

- dtype
  
Args:  

- A


```Mojo
sort[dtype: DType](owned A: Matrix[dtype], axis: Int) -> Matrix[dtype]
```  
Summary  
  
Sort the Matrix along the given axis.  
  
Parameters:  

- dtype
  
Args:  

- A
- axis

## max


```Mojo
max[dtype: DType](A: Matrix[dtype]) -> SIMD[dtype, 1]
```  
Summary  
  
Find max item. It is first flattened before sorting.  
  
Parameters:  

- dtype
  
Args:  

- A


```Mojo
max[dtype: DType](A: Matrix[dtype], axis: Int) -> Matrix[dtype]
```  
Summary  
  
Find max item along the given axis.  
  
Parameters:  

- dtype
  
Args:  

- A
- axis

## argmax


```Mojo
argmax[dtype: DType](A: Matrix[dtype]) -> SIMD[index, 1]
```  
Summary  
  
Index of the max. It is first flattened before sorting.  
  
Parameters:  

- dtype
  
Args:  

- A


```Mojo
argmax[dtype: DType](A: Matrix[dtype], axis: Int) -> Matrix[index]
```  
Summary  
  
Index of the max along the given axis.  
  
Parameters:  

- dtype
  
Args:  

- A
- axis

## min


```Mojo
min[dtype: DType](A: Matrix[dtype]) -> SIMD[dtype, 1]
```  
Summary  
  
Find min item. It is first flattened before sorting.  
  
Parameters:  

- dtype
  
Args:  

- A


```Mojo
min[dtype: DType](A: Matrix[dtype], axis: Int) -> Matrix[dtype]
```  
Summary  
  
Find min item along the given axis.  
  
Parameters:  

- dtype
  
Args:  

- A
- axis

## argmin


```Mojo
argmin[dtype: DType](A: Matrix[dtype]) -> SIMD[index, 1]
```  
Summary  
  
Index of the min. It is first flattened before sorting.  
  
Parameters:  

- dtype
  
Args:  

- A


```Mojo
argmin[dtype: DType](A: Matrix[dtype], axis: Int) -> Matrix[index]
```  
Summary  
  
Index of the min along the given axis.  
  
Parameters:  

- dtype
  
Args:  

- A
- axis
