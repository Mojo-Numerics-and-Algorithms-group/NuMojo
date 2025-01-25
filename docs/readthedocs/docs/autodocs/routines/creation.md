



# creation

##  Module Summary
  
Array creation routine.
## arange


```Mojo
arange[dtype: DType = float64](start: SIMD[dtype, 1], stop: SIMD[dtype, 1], step: SIMD[dtype, 1] = SIMD(1)) -> NDArray[dtype]
```  
Summary  
  
Function that computes a series of values starting from "start" to "stop" with given "step" size.  
  
Parameters:  

- dtype: Datatype of the output array. Defualt: `float64`
  
Args:  

- start: Scalar[dtype] - Start value.
- stop: Scalar[dtype]  - End value.
- step: Scalar[dtype]  - Step size between each element (default 1). Default: SIMD(1)


```Mojo
arange[dtype: DType = float64](stop: SIMD[dtype, 1]) -> NDArray[dtype]
```  
Summary  
  
(Overload) When start is 0 and step is 1.  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- stop

## linspace


```Mojo
linspace[dtype: DType = float64](start: SIMD[dtype, 1], stop: SIMD[dtype, 1], num: Int = 50, endpoint: Bool = True, parallel: Bool = False) -> NDArray[dtype]
```  
Summary  
  
Function that computes a series of linearly spaced values starting from "start" to "stop" with given size. Wrapper function for _linspace_serial, _linspace_parallel.  
  
Parameters:  

- dtype: Datatype of the output array. Defualt: `float64`
  
Args:  

- start: Start value.
- stop: End value.
- num: No of linearly spaced elements. Default: 50
- endpoint: Specifies whether to include endpoint in the final NDArray, defaults to True. Default: True
- parallel: Specifies whether the linspace should be calculated using parallelization, deafults to False. Default: False

## logspace


```Mojo
logspace[dtype: DType = float64](start: SIMD[dtype, 1], stop: SIMD[dtype, 1], num: Int, endpoint: Bool = True, base: SIMD[dtype, 1] = SIMD(#kgen.float_literal<10|1>), parallel: Bool = False) -> NDArray[dtype]
```  
Summary  
  
Generate a logrithmic spaced NDArray of `num` elements between `start` and `stop`. Wrapper function for _logspace_serial, _logspace_parallel functions.  
  
Parameters:  

- dtype: Datatype of the output array. Defualt: `float64`
  
Args:  

- start: The starting value of the NDArray.
- stop: The ending value of the NDArray.
- num: The number of elements in the NDArray.
- endpoint: Whether to include the `stop` value in the NDArray. Defaults to True. Default: True
- base: Base value of the logarithm, defaults to 10. Default: SIMD(#kgen.float_literal<10|1>)
- parallel: Specifies whether to calculate the logarithmic spaced values using parallelization. Default: False

## geomspace


```Mojo
geomspace[dtype: DType = float64](start: SIMD[dtype, 1], stop: SIMD[dtype, 1], num: Int, endpoint: Bool = True) -> NDArray[dtype]
```  
Summary  
  
Generate a NDArray of `num` elements between `start` and `stop` in a geometric series.  
  
Parameters:  

- dtype: Datatype of the input values. Defualt: `float64`
  
Args:  

- start: The starting value of the NDArray.
- stop: The ending value of the NDArray.
- num: The number of elements in the NDArray.
- endpoint: Whether to include the `stop` value in the NDArray. Defaults to True. Default: True

## empty


```Mojo
empty[dtype: DType = float64](shape: NDArrayShape) -> NDArray[dtype]
```  
Summary  
  
Generate an empty NDArray of given shape with arbitrary values.  
  
Parameters:  

- dtype: Datatype of the NDArray elements. Defualt: `float64`
  
Args:  

- shape: Shape of the NDArray.


```Mojo
empty[dtype: DType = float64](shape: List[Int]) -> NDArray[dtype]
```  
Summary  
  
Overload of function `empty` that reads a list of ints.  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- shape


```Mojo
empty[dtype: DType = float64](shape: VariadicList[Int]) -> NDArray[dtype]
```  
Summary  
  
Overload of function `empty` that reads a variadic list of ints.  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- shape

## empty_like


```Mojo
empty_like[dtype: DType = float64](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Generate an empty NDArray of the same shape as `array`.  
  
Parameters:  

- dtype: Datatype of the NDArray elements. Defualt: `float64`
  
Args:  

- array: NDArray to be used as a reference for the shape.

## eye


```Mojo
eye[dtype: DType = float64](N: Int, M: Int) -> NDArray[dtype]
```  
Summary  
  
Return a 2-D NDArray with ones on the diagonal and zeros elsewhere.  
  
Parameters:  

- dtype: Datatype of the NDArray elements. Defualt: `float64`
  
Args:  

- N: Number of rows in the matrix.
- M: Number of columns in the matrix.

## identity


```Mojo
identity[dtype: DType = float64](N: Int) -> NDArray[dtype]
```  
Summary  
  
Generate an identity matrix of size N x N.  
  
Parameters:  

- dtype: Datatype of the NDArray elements. Defualt: `float64`
  
Args:  

- N: Size of the matrix.

## ones


```Mojo
ones[dtype: DType = float64](shape: NDArrayShape) -> NDArray[dtype]
```  
Summary  
  
Generate a NDArray of ones with given shape filled with ones.  
  
Parameters:  

- dtype: Datatype of the NDArray. Defualt: `float64`
  
Args:  

- shape: Shape of the NDArray.


It calls the function `full`.


```Mojo
ones[dtype: DType = float64](shape: List[Int]) -> NDArray[dtype]
```  
Summary  
  
Overload of function `ones` that reads a list of ints.  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- shape


```Mojo
ones[dtype: DType = float64](shape: VariadicList[Int]) -> NDArray[dtype]
```  
Summary  
  
Overload of function `ones` that reads a variadic of ints.  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- shape

## ones_like


```Mojo
ones_like[dtype: DType = float64](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Generate a NDArray of the same shape as `a` filled with ones.  
  
Parameters:  

- dtype: Datatype of the NDArray. Defualt: `float64`
  
Args:  

- array: NDArray to be used as a reference for the shape.

## zeros


```Mojo
zeros[dtype: DType = float64](shape: NDArrayShape) -> NDArray[dtype]
```  
Summary  
  
Generate a NDArray of zeros with given shape.  
  
Parameters:  

- dtype: Datatype of the NDArray elements. Defualt: `float64`
  
Args:  

- shape: Shape of the NDArray.


It calls the function `full`.

of NDArray.

```Mojo
zeros[dtype: DType = float64](shape: List[Int]) -> NDArray[dtype]
```  
Summary  
  
Overload of function `zeros` that reads a list of ints.  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- shape


```Mojo
zeros[dtype: DType = float64](shape: VariadicList[Int]) -> NDArray[dtype]
```  
Summary  
  
Overload of function `zeros` that reads a variadic list of ints.  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- shape

## zeros_like


```Mojo
zeros_like[dtype: DType = float64](array: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Generate a NDArray of the same shape as `a` filled with zeros.  
  
Parameters:  

- dtype: Datatype of the NDArray elements. Defualt: `float64`
  
Args:  

- array: NDArray to be used as a reference for the shape.

## full


```Mojo
full[dtype: DType = float64](shape: NDArrayShape, fill_value: SIMD[dtype, 1], order: String = String("C")) -> NDArray[dtype]
```  
Summary  
  
Initialize an NDArray of certain shape fill it with a given value.  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- shape: Shape of the array.
- fill_value: Set all the values to this.
- order: Memory order C or F. Default: String("C")


Example:
    ```mojo
    import numojo as nm
    from numojo.prelude import *
    var a = nm.full(Shape(2,3,4), fill_value=10)
    ```

```Mojo
full[dtype: DType = float64](shape: List[Int], fill_value: SIMD[dtype, 1], order: String = String("C")) -> NDArray[dtype]
```  
Summary  
  
Overload of function `full` that reads a list of ints.  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- shape
- fill_value
- order Default: String("C")


```Mojo
full[dtype: DType = float64](shape: VariadicList[Int], fill_value: SIMD[dtype, 1], order: String = String("C")) -> NDArray[dtype]
```  
Summary  
  
Overload of function `full` that reads a variadic list of ints.  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- shape
- fill_value
- order Default: String("C")

## full_like


```Mojo
full_like[dtype: DType = float64](array: NDArray[dtype], fill_value: SIMD[dtype, 1], order: String = String("C")) -> NDArray[dtype]
```  
Summary  
  
Generate a NDArray of the same shape as `a` filled with `fill_value`.  
  
Parameters:  

- dtype: Datatype of the NDArray elements. Defualt: `float64`
  
Args:  

- array: NDArray to be used as a reference for the shape.
- fill_value: Value to fill the NDArray with.
- order: Memory order C or F. Default: String("C")

## diag


```Mojo
diag[dtype: DType = float64](v: NDArray[dtype], k: Int = 0) -> NDArray[dtype]
```  
Summary  
  
Extract a diagonal or construct a diagonal NDArray.  
  
Parameters:  

- dtype: Datatype of the NDArray elements. Defualt: `float64`
  
Args:  

- v: NDArray to extract the diagonal from.
- k: Diagonal offset. Default: 0

## diagflat


```Mojo
diagflat[dtype: DType = float64](mut v: NDArray[dtype], k: Int = 0) -> NDArray[dtype]
```  
Summary  
  
Generate a 2-D NDArray with the flattened input as the diagonal.  
  
Parameters:  

- dtype: Datatype of the NDArray elements. Defualt: `float64`
  
Args:  

- v: NDArray to be flattened and used as the diagonal.
- k: Diagonal offset. Default: 0

## tri


```Mojo
tri[dtype: DType = float64](N: Int, M: Int, k: Int = 0) -> NDArray[dtype]
```  
Summary  
  
Generate a 2-D NDArray with ones on and below the k-th diagonal.  
  
Parameters:  

- dtype: Datatype of the NDArray elements. Defualt: `float64`
  
Args:  

- N: Number of rows in the matrix.
- M: Number of columns in the matrix.
- k: Diagonal offset. Default: 0

## tril


```Mojo
tril[dtype: DType = float64](m: NDArray[dtype], k: Int = 0) -> NDArray[dtype]
```  
Summary  
  
Zero out elements above the k-th diagonal.  
  
Parameters:  

- dtype: Datatype of the NDArray elements. Defualt: `float64`
  
Args:  

- m: NDArray to be zeroed out.
- k: Diagonal offset. Default: 0

## triu


```Mojo
triu[dtype: DType = float64](m: NDArray[dtype], k: Int = 0) -> NDArray[dtype]
```  
Summary  
  
Zero out elements below the k-th diagonal.  
  
Parameters:  

- dtype: Datatype of the NDArray elements. Defualt: `float64`
  
Args:  

- m: NDArray to be zeroed out.
- k: Diagonal offset. Default: 0

## vander


```Mojo
vander[dtype: DType = float64](x: NDArray[dtype], N: Optional[Int] = Optional(None), increasing: Bool = False) -> NDArray[dtype]
```  
Summary  
  
Generate a Vandermonde matrix.  
  
Parameters:  

- dtype: Datatype of the NDArray elements. Defualt: `float64`
  
Args:  

- x: 1-D input array.
- N: Number of columns in the output. If N is not specified, a square array is returned. Default: Optional(None)
- increasing: Order of the powers of the columns. If True, the powers increase from left to right, if False (the default) they are reversed. Default: False

## astype


```Mojo
astype[dtype: DType, //, target: DType](a: NDArray[dtype]) -> NDArray[target]
```  
Summary  
  
Cast an NDArray to a different dtype.  
  
Parameters:  

- dtype: Data type of the input array, always inferred.
- target: Data type to cast the NDArray to.
  
Args:  

- a: NDArray to be casted.

## fromstring


```Mojo
fromstring[dtype: DType = float64](text: String, order: String = String("C")) -> NDArray[dtype]
```  
Summary  
  
NDArray initialization from string representation of an ndarray. The shape can be inferred from the string representation. The literals will be casted to the dtype of the NDArray.  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- text: String representation of an ndarray.
- order: Memory order C or F. Default: String("C")


Note:
StringLiteral is also allowed as input as it is coerced to String type
before it is passed into the function.

Example:
```
import numojo as nm

fn main() raises:
    var A = nm.fromstring[DType.int8]("[[[1,2],[3,4]],[[5,6],[7,8]]]")
    var B = nm.fromstring[DType.float16]("[[1,2,3,4],[5,6,7,8]]")
    var C = nm.fromstring[DType.float32]("[0.1, -2.3, 41.5, 19.29145, -199]")
    var D = nm.fromstring[DType.int32]("[0.1, -2.3, 41.5, 19.29145, -199]")

    print(A)
    print(B)
    print(C)
    print(D)
```

The output goes as follows. Note that the numbers are automatically
casted to the dtype of the NDArray.

```console
[[[     1       2       ]
 [     3       4       ]]
 [[     5       6       ]
 [     7       8       ]]]
3-D array  Shape: [2, 2, 2]  DType: int8

[[      1.0     2.0     3.0     4.0     ]
 [      5.0     6.0     7.0     8.0     ]]
2-D array  Shape: [2, 4]  DType: float16

[       0.10000000149011612     2.2999999523162842      41.5    19.291450500488281      199.0   ]
1-D array  Shape: [5]  DType: float32

[       0       2       41      19      199     ]
1-D array  Shape: [5]  DType: int32
```

## from_tensor


```Mojo
from_tensor[dtype: DType = float64](data: Tensor[dtype]) -> NDArray[dtype]
```  
Summary  
  
Create array from tensor.  
  
Parameters:  

- dtype: Datatype of the NDArray elements. Defualt: `float64`
  
Args:  

- data: Tensor.

## array


```Mojo
array[dtype: DType = float64](text: String, order: String = String("C")) -> NDArray[dtype]
```  
Summary  
  
This reload is an alias of `fromstring`.  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- text
- order Default: String("C")


```Mojo
array[dtype: DType = float64](data: List[SIMD[dtype, 1]], shape: List[Int], order: String = String("C")) -> NDArray[dtype]
```  
Summary  
  
Array creation with given data, shape and order.  
  
Parameters:  

- dtype: Datatype of the NDArray elements. Defualt: `float64`
  
Args:  

- data: List of data.
- shape: List of shape.
- order: Memory order C or F. Default: String("C")


Example:
    ```mojo
    import numojo as nm
    from numojo.prelude import *
    nm.array[f16](data=List[Scalar[f16]](1, 2, 3, 4), shape=List[Int](2, 2))
    ```


```Mojo
array[dtype: DType = float64](data: PythonObject, order: String = String("C")) -> NDArray[dtype]
```  
Summary  
  
Array creation with given data, shape and order.  
  
Parameters:  

- dtype: Datatype of the NDArray elements. Defualt: `float64`
  
Args:  

- data: A Numpy array (PythonObject).
- order: Memory order C or F. Default: String("C")


Example:
```mojo
import numojo as nm
from numojo.prelude import *
from python import Python
var np = Python.import_module("numpy")
var np_arr = np.array([1, 2, 3, 4])
A = nm.array[f16](data=np_arr, order="C")
```


```Mojo
array[dtype: DType = float64](data: Tensor[dtype]) -> NDArray[dtype]
```  
Summary  
  
Create array from tensor.  
  
Parameters:  

- dtype: Datatype of the NDArray elements. Defualt: `float64`
  
Args:  

- data: Tensor.


Example:
```mojo
import numojo as nm
from tensor import Tensor, TensorShape
from numojo.prelude import *

fn main() raises:
    height = 256
    width = 256
    channels = 3
    image = Tensor[DType.float32].rand(TensorShape(height, width, channels))
    print(image)
    print(nm.array(image))
```
