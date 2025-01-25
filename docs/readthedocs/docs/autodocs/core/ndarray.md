



# ndarray

##  Module Summary
  
Implements basic object methods for working with N-Dimensional Array.
## NDArray

### NDArray Summary
  
  
The N-dimensional array (NDArray).  

### Parent Traits
  

- Absable
- AnyType
- CollectionElement
- Copyable
- Movable
- Representable
- Sized
- Stringable
- UnknownDestructibility
- Writable

### Aliases
  
`width`: Vector size of the data type.
### Fields
  
  
* ndim `Int`  
    - Number of Dimensions.  
* shape `NDArrayShape`  
    - Size and shape of NDArray.  
* size `Int`  
    - Size of NDArray.  
* strides `NDArrayStrides`  
    - Contains offset, strides.  
* flags `Dict[String, Bool]`  
    - Information about the memory layout of the array.  

### Functions

#### __init__


```Mojo
__init__(out self, shape: NDArrayShape, order: String = String("C"))
```  
Summary  
  
Initialize an NDArray with given shape.  
  
Args:  

- self
- shape: Variadic shape.
- order: Memory order C or F. Default: String("C")


The memory is not filled with values.


```Mojo
__init__(out self, shape: List[Int], order: String = String("C"))
```  
Summary  
  
(Overload) Initialize an NDArray with given shape (list of integers).  
  
Args:  

- self
- shape: List of shape.
- order: Memory order C or F. Default: String("C")


```Mojo
__init__(out self, shape: VariadicList[Int], order: String = String("C"))
```  
Summary  
  
(Overload) Initialize an NDArray with given shape (variadic list of integers).  
  
Args:  

- self
- shape: Variadic List of shape.
- order: Memory order C or F. Default: String("C")


```Mojo
__init__(out self, shape: List[Int], offset: Int, strides: List[Int])
```  
Summary  
  
Extremely specific NDArray initializer.  
  
Args:  

- self
- shape
- offset
- strides


```Mojo
__init__(out self, shape: NDArrayShape, ref buffer: UnsafePointer[SIMD[dtype, 1]], offset: Int, strides: NDArrayStrides)
```  
Summary  
  
  
  
Args:  

- self
- shape
- buffer
- offset
- strides

#### __copyinit__


```Mojo
__copyinit__(out self, other: Self)
```  
Summary  
  
Copy other into self.  
  
Args:  

- self
- other


It is a deep copy. So the new array owns the data.
#### __moveinit__


```Mojo
__moveinit__(out self, owned existing: Self)
```  
Summary  
  
Move other into self.  
  
Args:  

- self
- existing

#### __del__


```Mojo
__del__(owned self)
```  
Summary  
  
  
  
Args:  

- self

#### __bool__


```Mojo
__bool__(self) -> Bool
```  
Summary  
  
If all true return true.  
  
Args:  

- self

#### __getitem__


```Mojo
__getitem__(self, index: Item) -> SIMD[dtype, 1]
```  
Summary  
  
Set the value at the index list.  
  
Args:  

- self
- index


```Mojo
__getitem__(self, idx: Int) -> Self
```  
Summary  
  
Retreive a slice of the array corresponding to the index at the first dimension.  
  
Args:  

- self
- idx


Example:
    `arr[1]` returns the second row of the array.

```Mojo
__getitem__(self, owned *slices: Slice) -> Self
```  
Summary  
  
Retreive slices of an array from variadic slices.  
  
Args:  

- self
- \*slices


Example:
    `arr[1:3, 2:4]` returns the corresponding sliced array (2 x 2).

```Mojo
__getitem__(self, owned slice_list: List[Slice]) -> Self
```  
Summary  
  
Retreive slices of an array from list of slices.  
  
Args:  

- self
- slice_list


Example:
    `arr[1:3, 2:4]` returns the corresponding sliced array (2 x 2).

```Mojo
__getitem__(self, owned *slices: Variant[Slice, Int]) -> Self
```  
Summary  
  
Get items by a series of either slices or integers.  
  
Args:  

- self
- \*slices: A series of either Slice or Int.


A decrease of dimensions may or may not happen when `__getitem__` is
called on an ndarray. An ndarray of X-D array can become Y-D array after
`__getitem__` where `Y <= X`.

Whether the dimension decerases or not depends on:
1. What types of arguments are passed into `__getitem__`.
2. The number of arguments that are passed in `__getitem__`.

PRINCIPAL: The number of dimensions to be decreased is determined by
the number of `Int` passed in `__getitem__`.

For example, `A` is a 10x10x10 ndarray (3-D). Then,

- `A[1, 2, 3]` leads to a 0-D array (scalar), since there are 3 integers.
- `A[1, 2]` leads to a 1-D array (vector), since there are 2 integers,
so the dimension decreases by 2.
- `A[1]` leads to a 2-D array (matrix), since there is 1 integer, so the
dimension decreases by 1.

The number of dimensions will not decrease when Slice is passed in
`__getitem__` or no argument is passed in for a certain dimension
(it is an implicit slide and a slide of all items will be used).

Take the same example `A` with 10x10x10 in shape. Then,

- `A[1:4, 2:5, 3:6]`, leads to a 3-D array (no decrease in dimension),
since there are 3 slices.
- `A[2:8]`, leads to a 3-D array (no decrease in dimension), since there
are 1 explicit slice and 2 implicit slices.

When there is a mixture of int and slices passed into `__getitem__`,
the number of integers will be the number of dimensions to be decreased.
Example,

- `A[1:4, 2, 2]`, leads to a 1-D array (vector), since there are 2
integers, so the dimension decreases by 2.

Note that, even though a slice contains one row, it does not reduce the
dimensions. Example,

- `A[1:2, 2:3, 3:4]`, leads to a 3-D array (no decrease in dimension),
since there are 3 slices.

Note that, when the number of integers equals to the number of
dimensions, the final outcome is an 0-D array instead of a number.
The user has to upack the 0-D array with the method`A.item(0)` to get the
corresponding number.
This behavior is different from numpy where the latter returns a number.

More examples for 1-D, 2-D, and 3-D arrays.

```console
A is a matrix
[[      -128    -95     65      -11     ]
[      8       -72     -116    45      ]
[      45      111     -30     4       ]
[      84      -120    -115    7       ]]
2-D array  Shape: [4, 4]  DType: int8

A[0]
[       -128    -95     65      -11     ]
1-D array  Shape: [4]  DType: int8

A[0, 1]
-95
0-D array  Shape: [0]  DType: int8

A[Slice(1,3)]
[[      8       -72     -116    45      ]
[      45      111     -30     4       ]]
2-D array  Shape: [2, 4]  DType: int8

A[1, Slice(2,4)]
[       -116    45      ]
1-D array  Shape: [2]  DType: int8

A[Slice(1,3), Slice(1,3)]
[[      -72     -116    ]
[      111     -30     ]]
2-D array  Shape: [2, 2]  DType: int8

A.item(0,1) as Scalar
-95

==============================
A is a vector
[       43      -127    -30     -111    ]
1-D array  Shape: [4]  DType: int8

A[0]
43
0-D array  Shape: [0]  DType: int8

A[Slice(1,3)]
[       -127    -30     ]
1-D array  Shape: [2]  DType: int8

A.item(0) as Scalar
43

==============================
A is a 3darray
[[[     -22     47      22      110     ]
[     88      6       -105    39      ]
[     -22     51      105     67      ]
[     -61     -116    60      -44     ]]
[[     33      65      125     -35     ]
[     -65     123     57      64      ]
[     38      -110    33      98      ]
[     -59     -17     68      -6      ]]
[[     -68     -58     -37     -86     ]
[     -4      101     104     -113    ]
[     103     1       4       -47     ]
[     124     -2      -60     -105    ]]
[[     114     -110    0       -30     ]
[     -58     105     7       -10     ]
[     112     -116    66      69      ]
[     83      -96     -124    48      ]]]
3-D array  Shape: [4, 4, 4]  DType: int8

A[0]
[[      -22     47      22      110     ]
[      88      6       -105    39      ]
[      -22     51      105     67      ]
[      -61     -116    60      -44     ]]
2-D array  Shape: [4, 4]  DType: int8

A[0, 1]
[       88      6       -105    39      ]
1-D array  Shape: [4]  DType: int8

A[0, 1, 2]
-105
0-D array  Shape: [0]  DType: int8

A[Slice(1,3)]
[[[     33      65      125     -35     ]
[     -65     123     57      64      ]
[     38      -110    33      98      ]
[     -59     -17     68      -6      ]]
[[     -68     -58     -37     -86     ]
[     -4      101     104     -113    ]
[     103     1       4       -47     ]
[     124     -2      -60     -105    ]]]
3-D array  Shape: [2, 4, 4]  DType: int8

A[1, Slice(2,4)]
[[      38      -110    33      98      ]
[      -59     -17     68      -6      ]]
2-D array  Shape: [2, 4]  DType: int8

A[Slice(1,3), Slice(1,3), 2]
[[      57      33      ]
[      104     4       ]]
2-D array  Shape: [2, 2]  DType: int8

A.item(0,1,2) as Scalar
-105
```


```Mojo
__getitem__(self, indices: NDArray[index]) -> Self
```  
Summary  
  
Get items from 0-th dimension of an ndarray of indices.  
  
Args:  

- self
- indices: Array of intable values.


If the original array is of shape (i,j,k) and
the indices array is of shape (l,m,n), then the output array
will be of shape (l,m,n,j,k).

Example:
```console
>>>var a = nm.arange[i8](6)
>>>print(a)
[       0       1       2       3       4       5       ]
1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True
>>>print(a[nm.array[isize]("[4, 2, 5, 1, 0, 2]")])
[       4       2       5       1       0       2       ]
1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True

var b = nm.arange[i8](12).reshape(Shape(2, 2, 3))
print(b)
[[[     0       1       2       ]
  [     3       4       5       ]]
 [[     6       7       8       ]
  [     9       10      11      ]]]
3-D array  Shape: [2, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
print(b[nm.array[isize]("[2, 0, 1]")])
[[[     0       0       0       ]
  [     0       67      95      ]]
 [[     0       1       2       ]
  [     3       4       5       ]]
 [[     6       7       8       ]
  [     9       10      11      ]]]
3-D array  Shape: [3, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
```


```Mojo
__getitem__(self, indices: List[Int]) -> Self
```  
Summary  
  
Get items from 0-th dimension of an array. It is an overload of `__getitem__(self, indices: NDArray[DType.index]) raises -> Self`.  
  
Args:  

- self
- indices: A list of Int.


Example:
```console
>>>var a = nm.arange[i8](6)
>>>print(a)
[       0       1       2       3       4       5       ]
1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True
>>>print(a[List[Int](4, 2, 5, 1, 0, 2)])
[       4       2       5       1       0       2       ]
1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True

var b = nm.arange[i8](12).reshape(Shape(2, 2, 3))
print(b)
[[[     0       1       2       ]
  [     3       4       5       ]]
 [[     6       7       8       ]
  [     9       10      11      ]]]
3-D array  Shape: [2, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
print(b[List[Int](2, 0, 1)])
[[[     0       0       0       ]
  [     0       67      95      ]]
 [[     0       1       2       ]
  [     3       4       5       ]]
 [[     6       7       8       ]
  [     9       10      11      ]]]
3-D array  Shape: [3, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
```


```Mojo
__getitem__(self, mask: NDArray[bool]) -> Self
```  
Summary  
  
Get item from an array according to a mask array.  
  
Args:  

- self
- mask: NDArray with Dtype.bool.


If array shape is equal to mask shape, it returns a flattened array of
the values where mask is True.

If array shape is not equal to mask shape, it returns items from the
0-th dimension of the array where mask is True.

Example:
```console
>>>var a = nm.arange[i8](6)
>>>print(a)
[       0       1       2       3       4       5       ]
1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True
>>>print(a[nm.array[boolean]("[1,0,1,1,0,1]")])
[       0       2       3       5       ]
1-D array  Shape: [4]  DType: int8  C-cont: True  F-cont: True  own data: True

var b = nm.arange[i8](12).reshape(Shape(2, 2, 3))
print(b)
[[[     0       1       2       ]
  [     3       4       5       ]]
 [[     6       7       8       ]
  [     9       10      11      ]]]
3-D array  Shape: [2, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
>>>print(b[nm.array[boolean]("[0,1]")])
[[[     6       7       8       ]
  [     9       10      11      ]]]
3-D array  Shape: [1, 2, 3]  DType: int8  C-cont: True  F-cont: True  own data: True
```


```Mojo
__getitem__(self, mask: List[Bool]) -> Self
```  
Summary  
  
Get items from 0-th dimension of an array according to mask. __getitem__(self, mask: NDArray[DType.bool]) raises -> Self.  
  
Args:  

- self
- mask: A list of boolean values.


Example:
```console
>>>var a = nm.arange[i8](6)
>>>print(a)
[       0       1       2       3       4       5       ]
1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True
>>>print(a[List[Bool](True, False, True, True, False, True)])
[       0       2       3       5       ]
1-D array  Shape: [4]  DType: int8  C-cont: True  F-cont: True  own data: True

var b = nm.arange[i8](12).reshape(Shape(2, 2, 3))
print(b)
[[[     0       1       2       ]
  [     3       4       5       ]]
 [[     6       7       8       ]
  [     9       10      11      ]]]
3-D array  Shape: [2, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
>>>print(b[List[Bool](False, True)])
[[[     6       7       8       ]
  [     9       10      11      ]]]
3-D array  Shape: [1, 2, 3]  DType: int8  C-cont: True  F-cont: True  own data: True
```

#### __setitem__


```Mojo
__setitem__(mut self, idx: Int, val: Self)
```  
Summary  
  
Set a slice of array with given array.  
  
Args:  

- self
- idx
- val


Example:
```mojo
import numojo as nm
var A = nm.random.rand[nm.i16](3, 2)
var B = nm.random.rand[nm.i16](3)
A[1:4] = B
```

```Mojo
__setitem__(mut self, index: Item, val: SIMD[dtype, 1])
```  
Summary  
  
Set the value at the index list.  
  
Args:  

- self
- index
- val


```Mojo
__setitem__(mut self, mask: NDArray[bool], value: SIMD[dtype, 1])
```  
Summary  
  
Set the value of the array at the indices where the mask is true.  
  
Args:  

- self
- mask
- value


```Mojo
__setitem__(mut self, *slices: Slice, *, val: Self)
```  
Summary  
  
Retreive slices of an array from variadic slices.  
  
Args:  

- self
- \*slices
- val


Example:
    `arr[1:3, 2:4]` returns the corresponding sliced array (2 x 2).

```Mojo
__setitem__(mut self, slices: List[Slice], val: Self)
```  
Summary  
  
Sets the slices of an array from list of slices and array.  
  
Args:  

- self
- slices
- val


Example:
```console
>>> var a = nm.arange[i8](16).reshape(Shape(4, 4))
print(a)
[[      0       1       2       3       ]
 [      4       5       6       7       ]
 [      8       9       10      11      ]
 [      12      13      14      15      ]]
2-D array  Shape: [4, 4]  DType: int8  C-cont: True  F-cont: False  own data: True
>>> a[2:4, 2:4] = a[0:2, 0:2]
print(a)
[[      0       1       2       3       ]
 [      4       5       6       7       ]
 [      8       9       0       1       ]
 [      12      13      4       5       ]]
2-D array  Shape: [4, 4]  DType: int8  C-cont: True  F-cont: False  own data: True
```

```Mojo
__setitem__(mut self, *slices: Variant[Slice, Int], *, val: Self)
```  
Summary  
  
Get items by a series of either slices or integers.  
  
Args:  

- self
- \*slices
- val


Example:
```console
>>> var a = nm.arange[i8](16).reshape(Shape(4, 4))
print(a)
[[      0       1       2       3       ]
 [      4       5       6       7       ]
 [      8       9       10      11      ]
 [      12      13      14      15      ]]
2-D array  Shape: [4, 4]  DType: int8  C-cont: True  F-cont: False  own data: True
>>> a[0, Slice(2, 4)] = a[3, Slice(0, 2)]
print(a)
[[      0       1       12      13      ]
 [      4       5       6       7       ]
 [      8       9       10      11      ]
 [      12      13      14      15      ]]
2-D array  Shape: [4, 4]  DType: int8  C-cont: True  F-cont: False  own data: True
```

```Mojo
__setitem__(self, index: NDArray[index], val: NDArray[dtype])
```  
Summary  
  
Returns the items of the array from an array of indices.  
  
Args:  

- self
- index
- val


Refer to `__getitem__(self, index: List[Int])`.

Example:
```console
> var X = nm.NDArray[nm.i8](3,random=True)
> print(X)
[       32      21      53      ]
1-D array  Shape: [3]  DType: int8
> print(X.argsort())
[       1       0       2       ]
1-D array  Shape: [3]  DType: index
> print(X[X.argsort()])
[       21      32      53      ]
1-D array  Shape: [3]  DType: int8
```

```Mojo
__setitem__(mut self, mask: NDArray[bool], val: Self)
```  
Summary  
  
Set the value of the array at the indices where the mask is true.  
  
Args:  

- self
- mask
- val


Example:
```
var A = numojo.core.NDArray[numojo.i16](6, random=True)
var mask = A > 0
print(A)
print(mask)
A[mask] = 0
print(A)
```
#### __neg__


```Mojo
__neg__(self) -> Self
```  
Summary  
  
Unary negative returns self unless boolean type.  
  
Args:  

- self


For bolean use `__invert__`(~)
#### __pos__


```Mojo
__pos__(self) -> Self
```  
Summary  
  
Unary positve returns self unless boolean type.  
  
Args:  

- self

#### __invert__


```Mojo
__invert__(self) -> Self
```  
Summary  
  
Element-wise inverse (~ or not), only for bools and integral types.  
  
Args:  

- self

#### __lt__


```Mojo
__lt__[OtherDtype: DType, ResultDType: DType = result[::DType,::DType]()](self, other: SIMD[OtherDtype, 1]) -> NDArray[bool]
```  
Summary  
  
Itemwise less than.  
  
Parameters:  

- OtherDtype
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__lt__[OtherDtype: DType, ResultDType: DType = result[::DType,::DType]()](self, other: NDArray[OtherDtype]) -> NDArray[bool]
```  
Summary  
  
Itemwise less than between scalar and Array.  
  
Parameters:  

- OtherDtype
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__lt__(self, other: SIMD[dtype, 1]) -> NDArray[bool]
```  
Summary  
  
Itemwise less than.  
  
Args:  

- self
- other


```Mojo
__lt__(self, other: Self) -> NDArray[bool]
```  
Summary  
  
Itemwise less than between scalar and Array.  
  
Args:  

- self
- other

#### __le__


```Mojo
__le__[OtherDtype: DType, ResultDType: DType = result[::DType,::DType]()](self, other: SIMD[OtherDtype, 1]) -> NDArray[bool]
```  
Summary  
  
Itemwise less than or equal to.  
  
Parameters:  

- OtherDtype
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__le__[OtherDtype: DType, ResultDType: DType = result[::DType,::DType]()](self, other: NDArray[OtherDtype]) -> NDArray[bool]
```  
Summary  
  
Itemwise less than or equal to between scalar and Array.  
  
Parameters:  

- OtherDtype
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__le__(self, other: SIMD[dtype, 1]) -> NDArray[bool]
```  
Summary  
  
Itemwise less than or equal to.  
  
Args:  

- self
- other


```Mojo
__le__(self, other: Self) -> NDArray[bool]
```  
Summary  
  
Itemwise less than or equal to between scalar and Array.  
  
Args:  

- self
- other

#### __eq__


```Mojo
__eq__[OtherDtype: DType, ResultDType: DType = result[::DType,::DType]()](self, other: NDArray[OtherDtype]) -> NDArray[bool]
```  
Summary  
  
Itemwise equivalence.  
  
Parameters:  

- OtherDtype
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__eq__[OtherDtype: DType, ResultDType: DType = result[::DType,::DType]()](self, other: SIMD[OtherDtype, 1]) -> NDArray[bool]
```  
Summary  
  
Itemwise equivalence.  
  
Parameters:  

- OtherDtype
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__eq__(self, other: Self) -> NDArray[bool]
```  
Summary  
  
Itemwise equivalence.  
  
Args:  

- self
- other


```Mojo
__eq__(self, other: SIMD[dtype, 1]) -> NDArray[bool]
```  
Summary  
  
Itemwise equivalence between scalar and Array.  
  
Args:  

- self
- other

#### __ne__


```Mojo
__ne__[OtherDtype: DType, ResultDType: DType = result[::DType,::DType]()](self, other: NDArray[OtherDtype]) -> NDArray[bool]
```  
Summary  
  
Itemwise nonequivelence.  
  
Parameters:  

- OtherDtype
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__ne__[OtherDtype: DType, ResultDType: DType = result[::DType,::DType]()](self, other: SIMD[OtherDtype, 1]) -> NDArray[bool]
```  
Summary  
  
Itemwise nonequivelence between scalar and Array.  
  
Parameters:  

- OtherDtype
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__ne__(self, other: SIMD[dtype, 1]) -> NDArray[bool]
```  
Summary  
  
Itemwise nonequivelence.  
  
Args:  

- self
- other


```Mojo
__ne__(self, other: Self) -> NDArray[bool]
```  
Summary  
  
Itemwise nonequivelence between scalar and Array.  
  
Args:  

- self
- other

#### __gt__


```Mojo
__gt__[OtherDtype: DType, ResultDType: DType = result[::DType,::DType]()](self, other: SIMD[OtherDtype, 1]) -> NDArray[bool]
```  
Summary  
  
Itemwise greater than.  
  
Parameters:  

- OtherDtype
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__gt__[OtherDtype: DType, ResultDType: DType = result[::DType,::DType]()](self, other: NDArray[OtherDtype]) -> NDArray[bool]
```  
Summary  
  
Itemwise greater than between scalar and Array.  
  
Parameters:  

- OtherDtype
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__gt__(self, other: SIMD[dtype, 1]) -> NDArray[bool]
```  
Summary  
  
Itemwise greater than.  
  
Args:  

- self
- other


```Mojo
__gt__(self, other: Self) -> NDArray[bool]
```  
Summary  
  
Itemwise greater than between scalar and Array.  
  
Args:  

- self
- other

#### __ge__


```Mojo
__ge__[OtherDtype: DType, ResultDType: DType = result[::DType,::DType]()](self, other: SIMD[OtherDtype, 1]) -> NDArray[bool]
```  
Summary  
  
Itemwise greater than or equal to.  
  
Parameters:  

- OtherDtype
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__ge__[OtherDtype: DType, ResultDType: DType = result[::DType,::DType]()](self, other: NDArray[OtherDtype]) -> NDArray[bool]
```  
Summary  
  
Itemwise less than or equal to between scalar and Array.  
  
Parameters:  

- OtherDtype
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__ge__(self, other: SIMD[dtype, 1]) -> NDArray[bool]
```  
Summary  
  
Itemwise greater than or equal to.  
  
Args:  

- self
- other


```Mojo
__ge__(self, other: Self) -> NDArray[bool]
```  
Summary  
  
Itemwise less than or equal to between scalar and Array.  
  
Args:  

- self
- other

#### __add__


```Mojo
__add__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, other: SIMD[OtherDType, 1]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `array + scalar`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__add__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, other: NDArray[OtherDType]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `array + array`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__add__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `array + scalar`.  
  
Args:  

- self
- other


```Mojo
__add__(self, other: Self) -> Self
```  
Summary  
  
Enables `array + array`.  
  
Args:  

- self
- other

#### __sub__


```Mojo
__sub__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, other: SIMD[OtherDType, 1]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `array - scalar`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__sub__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, other: NDArray[OtherDType]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `array - array`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__sub__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `array - scalar`.  
  
Args:  

- self
- other


```Mojo
__sub__(self, other: Self) -> Self
```  
Summary  
  
Enables `array - array`.  
  
Args:  

- self
- other

#### __mul__


```Mojo
__mul__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, other: SIMD[OtherDType, 1]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `array * scalar`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__mul__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, other: NDArray[OtherDType]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `array * array`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__mul__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `array * scalar`.  
  
Args:  

- self
- other


```Mojo
__mul__(self, other: Self) -> Self
```  
Summary  
  
Enables `array * array`.  
  
Args:  

- self
- other

#### __matmul__


```Mojo
__matmul__(self, other: Self) -> Self
```  
Summary  
  
  
  
Args:  

- self
- other

#### __truediv__


```Mojo
__truediv__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, other: SIMD[OtherDType, 1]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `array / scalar`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__truediv__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, other: NDArray[OtherDType]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `array / array`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__truediv__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `array / scalar`.  
  
Args:  

- self
- other


```Mojo
__truediv__(self, other: Self) -> Self
```  
Summary  
  
Enables `array / array`.  
  
Args:  

- self
- other

#### __floordiv__


```Mojo
__floordiv__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, other: SIMD[OtherDType, 1]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `array // scalar`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__floordiv__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, other: NDArray[OtherDType]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `array // array`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__floordiv__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `array // scalar`.  
  
Args:  

- self
- other


```Mojo
__floordiv__(self, other: Self) -> Self
```  
Summary  
  
Enables `array // array`.  
  
Args:  

- self
- other

#### __mod__


```Mojo
__mod__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, other: SIMD[OtherDType, 1]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `array % scalar`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__mod__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, other: NDArray[OtherDType]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `array % array`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__mod__(mut self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `array % scalar`.  
  
Args:  

- self
- other


```Mojo
__mod__(mut self, other: Self) -> Self
```  
Summary  
  
Enables `array % array`.  
  
Args:  

- self
- other

#### __pow__


```Mojo
__pow__(self, p: Int) -> Self
```  
Summary  
  
  
  
Args:  

- self
- p


```Mojo
__pow__(self, p: Self) -> Self
```  
Summary  
  
  
  
Args:  

- self
- p

#### __radd__


```Mojo
__radd__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, other: SIMD[OtherDType, 1]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `scalar + array`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__radd__(mut self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `scalar + array`.  
  
Args:  

- self
- other

#### __rsub__


```Mojo
__rsub__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, other: SIMD[OtherDType, 1]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `scalar - array`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__rsub__(mut self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `scalar - array`.  
  
Args:  

- self
- other

#### __rmul__


```Mojo
__rmul__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, other: SIMD[OtherDType, 1]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `scalar * array`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__rmul__(mut self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `scalar * array`.  
  
Args:  

- self
- other

#### __rtruediv__


```Mojo
__rtruediv__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, s: SIMD[OtherDType, 1]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `scalar / array`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- s


```Mojo
__rtruediv__(self, s: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `scalar / array`.  
  
Args:  

- self
- s

#### __rfloordiv__


```Mojo
__rfloordiv__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, other: SIMD[OtherDType, 1]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `scalar // array`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other


```Mojo
__rfloordiv__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `scalar // array`.  
  
Args:  

- self
- other

#### __rmod__


```Mojo
__rmod__(mut self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `scalar % array`.  
  
Args:  

- self
- other


```Mojo
__rmod__[OtherDType: DType, ResultDType: DType = result[::DType,::DType]()](self, other: SIMD[OtherDType, 1]) -> NDArray[ResultDType]
```  
Summary  
  
Enables `scalar % array`.  
  
Parameters:  

- OtherDType
- ResultDType Defualt: `result[::DType,::DType]()`
  
Args:  

- self
- other

#### __iadd__


```Mojo
__iadd__(mut self, other: SIMD[dtype, 1])
```  
Summary  
  
Enables `array += scalar`.  
  
Args:  

- self
- other


```Mojo
__iadd__(mut self, other: Self)
```  
Summary  
  
Enables `array *= array`.  
  
Args:  

- self
- other

#### __isub__


```Mojo
__isub__(mut self, other: SIMD[dtype, 1])
```  
Summary  
  
Enables `array -= scalar`.  
  
Args:  

- self
- other


```Mojo
__isub__(mut self, other: Self)
```  
Summary  
  
Enables `array -= array`.  
  
Args:  

- self
- other

#### __imul__


```Mojo
__imul__(mut self, other: SIMD[dtype, 1])
```  
Summary  
  
Enables `array *= scalar`.  
  
Args:  

- self
- other


```Mojo
__imul__(mut self, other: Self)
```  
Summary  
  
Enables `array *= array`.  
  
Args:  

- self
- other

#### __itruediv__


```Mojo
__itruediv__(mut self, s: SIMD[dtype, 1])
```  
Summary  
  
Enables `array /= scalar`.  
  
Args:  

- self
- s


```Mojo
__itruediv__(mut self, other: Self)
```  
Summary  
  
Enables `array /= array`.  
  
Args:  

- self
- other

#### __ifloordiv__


```Mojo
__ifloordiv__(mut self, s: SIMD[dtype, 1])
```  
Summary  
  
Enables `array //= scalar`.  
  
Args:  

- self
- s


```Mojo
__ifloordiv__(mut self, other: Self)
```  
Summary  
  
Enables `array //= array`.  
  
Args:  

- self
- other

#### __imod__


```Mojo
__imod__(mut self, other: SIMD[dtype, 1])
```  
Summary  
  
Enables `array %= scalar`.  
  
Args:  

- self
- other


```Mojo
__imod__(mut self, other: Self)
```  
Summary  
  
Enables `array %= array`.  
  
Args:  

- self
- other

#### __ipow__


```Mojo
__ipow__(mut self, p: Int)
```  
Summary  
  
  
  
Args:  

- self
- p

#### __int__


```Mojo
__int__(self) -> Int
```  
Summary  
  
Get Int representation of the array.  
  
Args:  

- self


Similar to Numpy, only 0-D arrays or length-1 arrays can be converted to
scalars.

Example:
```console
> var A = NDArray[dtype](6, random=True)
> print(int(A))

Unhandled exception caught during execution: Only 0-D arrays or length-1 arrays can be converted to scalars
mojo: error: execution exited with a non-zero result: 1

> var B = NDArray[dtype](1, 1, random=True)
> print(int(B))
14
```

#### __abs__


```Mojo
__abs__(self) -> Self
```  
Summary  
  
  
  
Args:  

- self

#### __str__


```Mojo
__str__(self) -> String
```  
Summary  
  
Enables str(array).  
  
Args:  

- self

#### write_to


```Mojo
write_to[W: Writer](self, mut writer: W)
```  
Summary  
  
  
  
Parameters:  

- W
  
Args:  

- self
- writer

#### __repr__


```Mojo
__repr__(self) -> String
```  
Summary  
  
Compute the "official" string representation of NDArray.  
  
Args:  

- self


You can construct the array using this representation.

An example is:
```console
>>>import numojo as nm
>>>var b = nm.arange[nm.f32](20).reshape(Shape(4, 5))
>>>print(repr(b))
numojo.array[f32](
'''
[[0.0, 1.0, 2.0, 3.0, 4.0]
 [5.0, 6.0, 7.0, 8.0, 9.0]
 [10.0, 11.0, 12.0, 13.0, 14.0]
 [15.0, 16.0, 17.0, 18.0, 19.0]]
'''
)
```
#### __len__


```Mojo
__len__(self) -> Int
```  
Summary  
  
Returns length of 0-th dimension.  
  
Args:  

- self

#### __iter__


```Mojo
__iter__(self) -> _NDArrayIter[self, dtype]
```  
Summary  
  
Iterate over elements of the NDArray and return sub-arrays as view.  
  
Args:  

- self


Example:
```
>>> var a = nm.random.arange[nm.i8](2 * 3 * 4).reshape(nm.Shape(2, 3, 4))
>>> for i in a:
...     print(i)
[[      0       1       2       3       ]
 [      4       5       6       7       ]
 [      8       9       10      11      ]]
2-D array  Shape: [3, 4]  DType: int8  C-cont: True  F-cont: False  own data: False
[[      12      13      14      15      ]
 [      16      17      18      19      ]
 [      20      21      22      23      ]]
2-D array  Shape: [3, 4]  DType: int8  C-cont: True  F-cont: False  own data: False
```.
#### __reversed__


```Mojo
__reversed__(self) -> _NDArrayIter[self, dtype, False]
```  
Summary  
  
Iterate backwards over elements of the NDArray, returning copied value.  
  
Args:  

- self

#### vdot


```Mojo
vdot(self, other: Self) -> SIMD[dtype, 1]
```  
Summary  
  
Inner product of two vectors.  
  
Args:  

- self
- other

#### mdot


```Mojo
mdot(self, other: Self) -> Self
```  
Summary  
  
Dot product of two matrix. Matrix A: M * N. Matrix B: N * L.  
  
Args:  

- self
- other

#### row


```Mojo
row(self, id: Int) -> Self
```  
Summary  
  
Get the ith row of the matrix.  
  
Args:  

- self
- id

#### col


```Mojo
col(self, id: Int) -> Self
```  
Summary  
  
Get the ith column of the matrix.  
  
Args:  

- self
- id

#### rdot


```Mojo
rdot(self, other: Self) -> Self
```  
Summary  
  
Dot product of two matrix. Matrix A: M * N. Matrix B: N * L.  
  
Args:  

- self
- other

#### num_elements


```Mojo
num_elements(self) -> Int
```  
Summary  
  
Function to retreive size (compatability).  
  
Args:  

- self

#### load


```Mojo
load(self, owned index: Int) -> SIMD[dtype, 1]
```  
Summary  
  
Safely retrieve i-th item from the underlying buffer.  
  
Args:  

- self
- index


`A.load(i)` differs from `A._buf.ptr[i]` due to boundary check.

Example:
```console
> array.load(15)
```
returns the item of index 15 from the array's data buffer.

Note that it does not checked against C-order or F-order.
```console
> # A is a 3x3 matrix, F-order (column-major)
> A.load(3)  # Row 0, Col 1
> A.item(3)  # Row 1, Col 0
```

```Mojo
load[width: Int = 1](self, index: Int) -> SIMD[dtype, width]
```  
Summary  
  
Safely loads a SIMD element of size `width` at `index` from the underlying buffer.  
  
Parameters:  

- width Defualt: `1`
  
Args:  

- self
- index


To bypass boundary checks, use `self._buf.ptr.load` directly.


```Mojo
load[width: Int = 1](self, *indices: Int) -> SIMD[dtype, width]
```  
Summary  
  
Safely loads SIMD element of size `width` at given variadic indices from the underlying buffer.  
  
Parameters:  

- width Defualt: `1`
  
Args:  

- self
- \*indices


To bypass boundary checks, use `self._buf.ptr.load` directly.

#### store


```Mojo
store(self, owned index: Int, val: SIMD[dtype, 1])
```  
Summary  
  
Safely store a scalar to i-th item of the underlying buffer.  
  
Args:  

- self
- index
- val


`A.store(i, a)` differs from `A._buf.ptr[i] = a` due to boundary check.

Example:
```console
> array.store(15, val = 100)
```
sets the item of index 15 of the array's data buffer to 100.

Note that it does not checked against C-order or F-order.

```Mojo
store[width: Int](mut self, index: Int, val: SIMD[dtype, width])
```  
Summary  
  
Safely stores SIMD element of size `width` at `index` of the underlying buffer.  
  
Parameters:  

- width
  
Args:  

- self
- index
- val


To bypass boundary checks, use `self._buf.ptr.store` directly.


```Mojo
store[width: Int = 1](mut self, *indices: Int, *, val: SIMD[dtype, width])
```  
Summary  
  
Safely stores SIMD element of size `width` at given variadic indices of the underlying buffer.  
  
Parameters:  

- width Defualt: `1`
  
Args:  

- self
- \*indices
- val


To bypass boundary checks, use `self._buf.ptr.store` directly.

#### T


```Mojo
T(self, axes: List[Int]) -> Self
```  
Summary  
  
Transpose array of any number of dimensions according to arbitrary permutation of the axes.  
  
Args:  

- self
- axes


If `axes` is not given, it is equal to flipping the axes.

Defined in `numojo.routines.manipulation.transpose`.

```Mojo
T(self) -> Self
```  
Summary  
  
(overload) Transpose the array when `axes` is not given. If `axes` is not given, it is equal to flipping the axes. See docstring of `transpose`.  
  
Args:  

- self


Defined in `numojo.routines.manipulation.transpose`.
#### all


```Mojo
all(self) -> Bool
```  
Summary  
  
If all true return true.  
  
Args:  

- self

#### any


```Mojo
any(self) -> Bool
```  
Summary  
  
True if any true.  
  
Args:  

- self

#### argmax


```Mojo
argmax(self) -> Int
```  
Summary  
  
Get location in pointer of max value.  
  
Args:  

- self

#### argmin


```Mojo
argmin(self) -> Int
```  
Summary  
  
Get location in pointer of min value.  
  
Args:  

- self

#### argsort


```Mojo
argsort(self) -> NDArray[index]
```  
Summary  
  
Sort the NDArray and return the sorted indices.  
  
Args:  

- self


See `numojo.routines.sorting.argsort()`.

#### astype


```Mojo
astype[target: DType](self) -> NDArray[target]
```  
Summary  
  
Convert type of array.  
  
Parameters:  

- target
  
Args:  

- self

#### copy


```Mojo
copy(self) -> Self
```  
Summary  
  
Returns a copy of the array that owns the data. The returned array will be continuous in memory.  
  
Args:  

- self

#### cumprod


```Mojo
cumprod(self) -> Self
```  
Summary  
  
Returns cumprod of all items of an array. The array is flattened before cumprod.  
  
Args:  

- self


```Mojo
cumprod(self, axis: Int) -> Self
```  
Summary  
  
Returns cumprod of array by axis.  
  
Args:  

- self
- axis: Axis.

#### cumsum


```Mojo
cumsum(self) -> Self
```  
Summary  
  
Returns cumsum of all items of an array. The array is flattened before cumsum.  
  
Args:  

- self


```Mojo
cumsum(self, axis: Int) -> Self
```  
Summary  
  
Returns cumsum of array by axis.  
  
Args:  

- self
- axis: Axis.

#### diagonal


```Mojo
diagonal(self)
```  
Summary  
  
  
  
Args:  

- self

#### fill


```Mojo
fill(mut self, val: SIMD[dtype, 1])
```  
Summary  
  
Fill all items of array with value.  
  
Args:  

- self
- val

#### flatten


```Mojo
flatten(self, order: String = String("C")) -> Self
```  
Summary  
  
Return a copy of the array collapsed into one dimension.  
  
Args:  

- self
- order: A NDArray. Default: String("C")

#### item


```Mojo
item(self, owned index: Int) -> ref [MutableAnyOrigin] SIMD[dtype, 1]
```  
Summary  
  
Return the scalar at the coordinates.  
  
Args:  

- self
- index: Index of item, counted in row-major way.


If one index is given, get the i-th item of the array (not buffer).
It first scans over the first row, even it is a colume-major array.

If more than one index is given, the length of the indices must match
the number of dimensions of the array.

Example:
```console
>>> var A = nm.random.randn[nm.f16](2, 2, 2)
>>> A = A.reshape(A.shape, order="F")
>>> print(A)
[[[     0.2446289       0.5419922       ]
  [     0.09643555      -0.90722656     ]]
 [[     1.1806641       0.24389648      ]
  [     0.5234375       1.0390625       ]]]
3-D array  Shape: [2, 2, 2]  DType: float16  order: F
>>> for i in range(A.size):
...     print(A.item(i))
0.2446289
0.5419922
0.09643555
-0.90722656
1.1806641
0.24389648
0.5234375
1.0390625
>>> print(A.item(0, 1, 1))
-0.90722656
```.

```Mojo
item(self, *index: Int) -> ref [MutableAnyOrigin] SIMD[dtype, 1]
```  
Summary  
  
Return the scalar at the coordinates.  
  
Args:  

- self
- \*index: The coordinates of the item.


If one index is given, get the i-th item of the array (not buffer).
It first scans over the first row, even it is a colume-major array.

If more than one index is given, the length of the indices must match
the number of dimensions of the array.

Example:
```
>>> var A = nm.random.randn[nm.f16](2, 2, 2)
>>> A = A.reshape(A.shape, order="F")
>>> print(A)
[[[     0.2446289       0.5419922       ]
  [     0.09643555      -0.90722656     ]]
 [[     1.1806641       0.24389648      ]
  [     0.5234375       1.0390625       ]]]
3-D array  Shape: [2, 2, 2]  DType: float16  order: F
>>> print(A.item(0, 1, 1))
-0.90722656
```.
#### itemset


```Mojo
itemset(mut self, index: Variant[Int, List[Int]], item: SIMD[dtype, 1])
```  
Summary  
  
Set the scalar at the coordinates.  
  
Args:  

- self
- index: The coordinates of the item. Can either be `Int` or `List[Int]`. If `Int` is passed, it is the index of i-th item of the whole array. If `List[Int]` is passed, it is the coordinate of the item.
- item: The scalar to be set.


Note:
    This is similar to `numpy.ndarray.itemset`.
    The difference is that we takes in `List[Int]`, but numpy takes in a tuple.

An example goes as follows.

```
import numojo as nm

fn main() raises:
    var A = nm.zeros[nm.i16](3, 3)
    print(A)
    A.itemset(5, 256)
    print(A)
    A.itemset(List(1,1), 1024)
    print(A)
```
```console
[[      0       0       0       ]
 [      0       0       0       ]
 [      0       0       0       ]]
2-D array  Shape: [3, 3]  DType: int16
[[      0       0       0       ]
 [      0       0       256     ]
 [      0       0       0       ]]
2-D array  Shape: [3, 3]  DType: int16
[[      0       0       0       ]
 [      0       1024    256     ]
 [      0       0       0       ]]
2-D array  Shape: [3, 3]  DType: int16
```
#### max


```Mojo
max(self, axis: Int = 0) -> Self
```  
Summary  
  
Max on axis.  
  
Args:  

- self
- axis Default: 0

#### min


```Mojo
min(self, axis: Int = 0) -> Self
```  
Summary  
  
Min on axis.  
  
Args:  

- self
- axis Default: 0

#### mean


```Mojo
mean(self, axis: Int) -> Self
```  
Summary  
  
Mean of array elements over a given axis. Args:     array: NDArray.     axis: The axis along which the mean is performed. Returns:     An NDArray.  
  
Args:  

- self
- axis


```Mojo
mean(self) -> SIMD[dtype, 1]
```  
Summary  
  
Cumulative mean of a array.  
  
Args:  

- self

#### nditer


```Mojo
nditer(self) -> _NDIter[self, dtype]
```  
Summary  
  
(Overload) Return an iterator yielding the array elements according to the memory layout of the array.  
  
Args:  

- self


```console
>>>var a = nm.random.rand[i8](2, 3, min=0, max=100)
>>>print(a)
[[      37      8       25      ]
 [      25      2       57      ]]
2-D array  (2,3)  DType: int8  C-cont: True  F-cont: False  own data: True
>>>for i in a.nditer():
...    print(i, end=" ")
37 8 25 25 2 57
```

```Mojo
nditer(self, order: String) -> _NDIter[self, dtype]
```  
Summary  
  
Return an iterator yielding the array elements according to the order.  
  
Args:  

- self
- order


```console
>>>var a = nm.random.rand[i8](2, 3, min=0, max=100)
>>>print(a)
[[      37      8       25      ]
 [      25      2       57      ]]
2-D array  (2,3)  DType: int8  C-cont: True  F-cont: False  own data: True
>>>for i in a.nditer():
...    print(i, end=" ")
37 8 25 25 2 57
```
#### prod


```Mojo
prod(self) -> SIMD[dtype, 1]
```  
Summary  
  
Product of all array elements. Returns:     Scalar.  
  
Args:  

- self


```Mojo
prod(self, axis: Int) -> Self
```  
Summary  
  
Product of array elements over a given axis. Args:     axis: The axis along which the product is performed. Returns:     An NDArray.  
  
Args:  

- self
- axis

#### reshape


```Mojo
reshape(self, shape: NDArrayShape, order: String = String("C")) -> Self
```  
Summary  
  
Returns an array of the same data with a new shape.  
  
Args:  

- self
- shape: Shape of returned array.
- order: Order of the array - Row major `C` or Column major `F`. Default: String("C")

#### resize


```Mojo
resize(mut self, shape: NDArrayShape)
```  
Summary  
  
In-place change shape and size of array.  
  
Args:  

- self
- shape: Shape after resize.


Notes:
To returns a new array, use `reshape`.

#### round


```Mojo
round(self) -> Self
```  
Summary  
  
Rounds the elements of the array to a whole number.  
  
Args:  

- self

#### sort


```Mojo
sort(mut self)
```  
Summary  
  
Sort NDArray using quick sort method. It is not guaranteed to be unstable.  
  
Args:  

- self


When no axis is given, the array is flattened before sorting.

See `numojo.sorting.sort` for more information.

```Mojo
sort(mut self, owned axis: Int)
```  
Summary  
  
Sort NDArray along the given axis using quick sort method. It is not guaranteed to be unstable.  
  
Args:  

- self
- axis


When no axis is given, the array is flattened before sorting.

See `numojo.sorting.sort` for more information.
#### sum


```Mojo
sum(self) -> SIMD[dtype, 1]
```  
Summary  
  
Sum of all array elements. Returns:     Scalar.  
  
Args:  

- self


```Mojo
sum(self, axis: Int) -> Self
```  
Summary  
  
Sum of array elements over a given axis. Args:     axis: The axis along which the sum is performed. Returns:     An NDArray.  
  
Args:  

- self
- axis

#### tolist


```Mojo
tolist(self) -> List[SIMD[dtype, 1]]
```  
Summary  
  
Convert NDArray to a 1-D List.  
  
Args:  

- self

#### to_numpy


```Mojo
to_numpy(self) -> PythonObject
```  
Summary  
  
Convert to a numpy array.  
  
Args:  

- self

#### to_tensor


```Mojo
to_tensor(self) -> Tensor[dtype]
```  
Summary  
  
Convert array to tensor of the same dtype.  
  
Args:  

- self


```mojo
import numojo as nm
from numojo.prelude import *

fn main() raises:
    var a = nm.random.randn[f16](2, 3, 4)
    print(a)
    print(a.to_tensor())

    var b = nm.array[i8]("[[1, 2, 3], [4, 5, 6]]")
    print(b)
    print(b.to_tensor())

    var c = nm.array[boolean]("[[1,0], [0,1]]")
    print(c)
    print(c.to_tensor())
```
#### trace


```Mojo
trace(self, offset: Int = 0, axis1: Int = 0, axis2: Int = 1) -> Self
```  
Summary  
  
Computes the trace of a ndarray.  
  
Args:  

- self
- offset: Offset of the diagonal from the main diagonal. Default: 0
- axis1: First axis. Default: 0
- axis2: Second axis. Default: 1

#### unsafe_ptr


```Mojo
unsafe_ptr(self) -> UnsafePointer[SIMD[dtype, 1]]
```  
Summary  
  
Retreive pointer without taking ownership.  
  
Args:  

- self
