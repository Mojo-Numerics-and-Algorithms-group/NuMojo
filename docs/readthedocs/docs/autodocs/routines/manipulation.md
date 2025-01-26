



# manipulation

##  Module Summary
  
Array manipulation routines.
## copyto


```Mojo
copyto()
```  
Summary  
  
  

## ndim


```Mojo
ndim[dtype: DType](array: NDArray[dtype]) -> Int
```  
Summary  
  
Returns the number of dimensions of the NDArray.  
  
Parameters:  

- dtype
  
Args:  

- array: A NDArray.

## shape


```Mojo
shape[dtype: DType](array: NDArray[dtype]) -> NDArrayShape
```  
Summary  
  
Returns the shape of the NDArray.  
  
Parameters:  

- dtype
  
Args:  

- array: A NDArray.

## size


```Mojo
size[dtype: DType](array: NDArray[dtype], axis: Int) -> Int
```  
Summary  
  
Returns the size of the NDArray.  
  
Parameters:  

- dtype
  
Args:  

- array: A NDArray.
- axis: The axis to get the size of.

## reshape


```Mojo
reshape[dtype: DType](owned A: NDArray[dtype], shape: NDArrayShape, order: String = String("C")) -> NDArray[dtype]
```  
Summary  
  
    Returns an array of the same data with a new shape.  
  
Parameters:  

- dtype
  
Args:  

- A: A NDArray.
- shape: New shape.
- order: "C" or "F". Read in this order from the original array and write in this order into the new array. Default: String("C")

## ravel


```Mojo
ravel[dtype: DType](owned A: NDArray[dtype], order: String = String("C")) -> NDArray[dtype]
```  
Summary  
  
Returns the raveled version of the NDArray.  
  
Parameters:  

- dtype
  
Args:  

- A: NDArray.
- order: The order to flatten the array. Default: String("C")


Return:
    A contiguous flattened array.
## transpose


```Mojo
transpose[dtype: DType](A: NDArray[dtype], axes: List[Int]) -> NDArray[dtype]
```  
Summary  
  
Transpose array of any number of dimensions according to arbitrary permutation of the axes.  
  
Parameters:  

- dtype
  
Args:  

- A
- axes


If `axes` is not given, it is equal to flipping the axes.
```mojo
import numojo as nm
var A = nm.random.rand(2,3,4,5)
print(nm.transpose(A))  # A is a 4darray.
print(nm.transpose(A, axes=List(3,2,1,0)))
```

Examples.
```mojo
import numojo as nm
# A is a 2darray
print(nm.transpose(A, axes=List(0, 1)))  # equal to transpose of matrix
# A is a 3darray
print(nm.transpose(A, axes=List(2, 1, 0)))  # transpose 0-th and 2-th dimensions
```

```Mojo
transpose[dtype: DType](A: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
(overload) Transpose the array when `axes` is not given. If `axes` is not given, it is equal to flipping the axes. See docstring of `transpose`.  
  
Parameters:  

- dtype
  
Args:  

- A


```Mojo
transpose[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]
```  
Summary  
  
Transpose of matrix.  
  
Parameters:  

- dtype
  
Args:  

- A

## flip


```Mojo
flip[dtype: DType](owned A: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Returns flipped array and keep the shape.  
  
Parameters:  

- dtype: DType.
  
Args:  

- A: A NDArray.


```Mojo
flip[dtype: DType](owned A: NDArray[dtype], owned axis: Int) -> NDArray[dtype]
```  
Summary  
  
Returns flipped array along the given axis.  
  
Parameters:  

- dtype: DType.
  
Args:  

- A: A NDArray.
- axis: Axis along which to flip.
