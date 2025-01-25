



# complex_ndarray

##  Module Summary
  
Implements N-Dimensional Complex Array Last updated: 2025-01-24
## ComplexNDArray

### ComplexNDArray Summary
  
  
Represents a Complex N-Dimensional Array.  

### Parent Traits
  

- AnyType
- CollectionElement
- Copyable
- Movable
- Representable
- Sized
- Stringable
- UnknownDestructibility
- Writable

### Fields
  
  
* ndim `Int`  
    - Number of Dimensions.  
* shape `NDArrayShape`  
    - Size and shape of ComplexNDArray.  
* size `Int`  
    - Size of ComplexNDArray.  
* strides `NDArrayStrides`  
    - Contains offset, strides.  
* flags `Dict[String, Bool]`  
    - Information about the memory layout of the array.  

### Functions

#### __init__


```Mojo
__init__(out self, owned re: NDArray[dtype], owned im: NDArray[dtype])
```  
Summary  
  
  
  
Args:  

- self
- re
- im


```Mojo
__init__(out self, shape: NDArrayShape, order: String = String("C"))
```  
Summary  
  
Initialize a ComplexNDArray with given shape.  
  
Args:  

- self
- shape: Variadic shape.
- order: Memory order C or F. Default: String("C")


The memory is not filled with values.

Example:
```mojo
import numojo as nm
var A = nm.ComplexNDArray[cf32](Shape(2,3,4))
```

```Mojo
__init__(out self, shape: List[Int], order: String = String("C"))
```  
Summary  
  
(Overload) Initialize a ComplexNDArray with given shape (list of integers).  
  
Args:  

- self
- shape: List of shape.
- order: Memory order C or F. Default: String("C")


```Mojo
__init__(out self, shape: VariadicList[Int], order: String = String("C"))
```  
Summary  
  
(Overload) Initialize a ComplexNDArray with given shape (variadic list of integers).  
  
Args:  

- self
- shape: Variadic List of shape.
- order: Memory order C or F. Default: String("C")


```Mojo
__init__(out self, shape: List[Int], offset: Int, strides: List[Int])
```  
Summary  
  
Extremely specific ComplexNDArray initializer.  
  
Args:  

- self
- shape
- offset
- strides


```Mojo
__init__(out self, shape: NDArrayShape, ref buffer_re: UnsafePointer[SIMD[dtype, 1]], ref buffer_im: UnsafePointer[SIMD[dtype, 1]], offset: Int, strides: NDArrayStrides)
```  
Summary  
  
Extremely specific ComplexNDArray initializer.  
  
Args:  

- self
- shape
- buffer_re
- buffer_im
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

#### __moveinit__


```Mojo
__moveinit__(out self, owned existing: Self)
```  
Summary  
  
Move other into self.  
  
Args:  

- self
- existing

#### __getitem__


```Mojo
__getitem__(self, idx: Int) -> Self
```  
Summary  
  
Retreive a slice of the ComplexNDArray corresponding to the index at the first dimension.  
  
Args:  

- self
- idx


Example:
    `arr[1]` returns the second row of the ComplexNDArray.

```Mojo
__getitem__(self, index: Item) -> ComplexSIMD[cdtype, dtype=dtype]
```  
Summary  
  
Get the value at the index list.  
  
Args:  

- self
- index


```Mojo
__getitem__(self, owned *slices: Slice) -> Self
```  
Summary  
  
Retreive slices of a ComplexNDArray from variadic slices.  
  
Args:  

- self
- \*slices


Example:
    `arr[1:3, 2:4]` returns the corresponding sliced ComplexNDArray (2 x 2).

```Mojo
__getitem__(self, owned slice_list: List[Slice]) -> Self
```  
Summary  
  
Retreive slices of a ComplexNDArray from list of slices.  
  
Args:  

- self
- slice_list


Example:
    `arr[1:3, 2:4]` returns the corresponding sliced ComplexNDArray (2 x 2).

```Mojo
__getitem__(self, owned *slices: Variant[Slice, Int]) -> Self
```  
Summary  
  
Get items by a series of either slices or integers.  
  
Args:  

- self
- \*slices: A series of either Slice or Int.


```Mojo
__getitem__(self, index: List[Int]) -> Self
```  
Summary  
  
Get items of ComplexNDArray from a list of indices.  
  
Args:  

- self
- index: List[Int].


It always gets the first dimension.
```


```Mojo
__getitem__(self, index: NDArray[index]) -> Self
```  
Summary  
  
Get items of ComplexNDArray from an array of indices.  
  
Args:  

- self
- index


Refer to `__getitem__(self, index: List[Int])`.

```Mojo
__getitem__(self, mask: NDArray[bool]) -> Self
```  
Summary  
  
Get items of ComplexNDArray corresponding to a mask.  
  
Args:  

- self
- mask: NDArray with Dtype.bool.


Example:
    ```
    var A = numojo.core.NDArray[numojo.i16](6, random=True)
    var mask = A > 0
    print(A)
    print(mask)
    print(A[mask])
    ```

#### __setitem__


```Mojo
__setitem__(mut self, idx: Int, val: Self)
```  
Summary  
  
Set a slice of ComplexNDArray with given ComplexNDArray.  
  
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
__setitem__(mut self, index: Item, val: ComplexSIMD[cdtype, dtype=dtype])
```  
Summary  
  
Set the value at the index list.  
  
Args:  

- self
- index
- val


```Mojo
__setitem__(mut self, mask: Self, value: ComplexSIMD[cdtype, dtype=dtype])
```  
Summary  
  
Set the value of the array at the indices where the mask is true.  
  
Args:  

- self
- mask
- value


```Mojo
__setitem__(mut self, owned *slices: Slice, *, val: Self)
```  
Summary  
  
Retreive slices of an ComplexNDArray from variadic slices.  
  
Args:  

- self
- \*slices
- val


Example:
    `arr[1:3, 2:4]` returns the corresponding sliced ComplexNDArray (2 x 2).

```Mojo
__setitem__(mut self, owned slices: List[Slice], val: Self)
```  
Summary  
  
Sets the slices of an ComplexNDArray from list of slices and ComplexNDArray.  
  
Args:  

- self
- slices
- val


Example:
    `arr[1:3, 2:4]` returns the corresponding sliced ComplexNDArray (2 x 2).

```Mojo
__setitem__(self, index: NDArray[index], val: Self)
```  
Summary  
  
Returns the items of the ComplexNDArray from an array of indices.  
  
Args:  

- self
- index
- val


Refer to `__getitem__(self, index: List[Int])`.

```Mojo
__setitem__(mut self, mask: Self, val: Self)
```  
Summary  
  
Set the value of the ComplexNDArray at the indices where the mask is true.  
  
Args:  

- self
- mask
- val

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
  
Unary positive returns self unless boolean type.  
  
Args:  

- self

#### __eq__


```Mojo
__eq__(self, other: Self) -> NDArray[bool]
```  
Summary  
  
Itemwise equivalence.  
  
Args:  

- self
- other


```Mojo
__eq__(self, other: ComplexSIMD[cdtype, dtype=dtype]) -> NDArray[bool]
```  
Summary  
  
Itemwise equivalence between scalar and ComplexNDArray.  
  
Args:  

- self
- other

#### __ne__


```Mojo
__ne__(self, other: Self) -> NDArray[bool]
```  
Summary  
  
Itemwise non-equivalence.  
  
Args:  

- self
- other


```Mojo
__ne__(self, other: ComplexSIMD[cdtype, dtype=dtype]) -> NDArray[bool]
```  
Summary  
  
Itemwise non-equivalence between scalar and ComplexNDArray.  
  
Args:  

- self
- other

#### __add__


```Mojo
__add__(self, other: ComplexSIMD[cdtype, dtype=dtype]) -> Self
```  
Summary  
  
Enables `ComplexNDArray + ComplexSIMD`.  
  
Args:  

- self
- other


```Mojo
__add__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `ComplexNDArray + Scalar`.  
  
Args:  

- self
- other


```Mojo
__add__(self, other: Self) -> Self
```  
Summary  
  
Enables `ComplexNDArray + ComplexNDArray`.  
  
Args:  

- self
- other


```Mojo
__add__(self, other: NDArray[dtype]) -> Self
```  
Summary  
  
Enables `ComplexNDArray + NDArray`.  
  
Args:  

- self
- other

#### __sub__


```Mojo
__sub__(self, other: ComplexSIMD[cdtype, dtype=dtype]) -> Self
```  
Summary  
  
Enables `ComplexNDArray - ComplexSIMD`.  
  
Args:  

- self
- other


```Mojo
__sub__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `ComplexNDArray - Scalar`.  
  
Args:  

- self
- other


```Mojo
__sub__(self, other: Self) -> Self
```  
Summary  
  
Enables `ComplexNDArray - ComplexNDArray`.  
  
Args:  

- self
- other


```Mojo
__sub__(self, other: NDArray[dtype]) -> Self
```  
Summary  
  
Enables `ComplexNDArray - NDArray`.  
  
Args:  

- self
- other

#### __mul__


```Mojo
__mul__(self, other: ComplexSIMD[cdtype, dtype=dtype]) -> Self
```  
Summary  
  
Enables `ComplexNDArray * ComplexSIMD`.  
  
Args:  

- self
- other


```Mojo
__mul__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `ComplexNDArray * Scalar`.  
  
Args:  

- self
- other


```Mojo
__mul__(self, other: Self) -> Self
```  
Summary  
  
Enables `ComplexNDArray * ComplexNDArray`.  
  
Args:  

- self
- other


```Mojo
__mul__(self, other: NDArray[dtype]) -> Self
```  
Summary  
  
Enables `ComplexNDArray * NDArray`.  
  
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
__truediv__(self, other: ComplexSIMD[cdtype, dtype=dtype]) -> Self
```  
Summary  
  
Enables `ComplexNDArray / ComplexSIMD`.  
  
Args:  

- self
- other


```Mojo
__truediv__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `ComplexNDArray / ComplexSIMD`.  
  
Args:  

- self
- other


```Mojo
__truediv__(self, other: Self) -> Self
```  
Summary  
  
Enables `ComplexNDArray / ComplexNDArray`.  
  
Args:  

- self
- other


```Mojo
__truediv__(self, other: NDArray[dtype]) -> Self
```  
Summary  
  
Enables `ComplexNDArray / NDArray`.  
  
Args:  

- self
- other

#### __radd__


```Mojo
__radd__(mut self, other: ComplexSIMD[cdtype, dtype=dtype]) -> Self
```  
Summary  
  
Enables `ComplexSIMD + ComplexNDArray`.  
  
Args:  

- self
- other


```Mojo
__radd__(mut self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `Scalar + ComplexNDArray`.  
  
Args:  

- self
- other


```Mojo
__radd__(mut self, other: NDArray[dtype]) -> Self
```  
Summary  
  
Enables `NDArray + ComplexNDArray`.  
  
Args:  

- self
- other

#### __rsub__


```Mojo
__rsub__(mut self, other: ComplexSIMD[cdtype, dtype=dtype]) -> Self
```  
Summary  
  
Enables `ComplexSIMD - ComplexNDArray`.  
  
Args:  

- self
- other


```Mojo
__rsub__(mut self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `Scalar - ComplexNDArray`.  
  
Args:  

- self
- other


```Mojo
__rsub__(mut self, other: NDArray[dtype]) -> Self
```  
Summary  
  
Enables `NDArray - ComplexNDArray`.  
  
Args:  

- self
- other

#### __rmul__


```Mojo
__rmul__(self, other: ComplexSIMD[cdtype, dtype=dtype]) -> Self
```  
Summary  
  
Enables `ComplexSIMD * ComplexNDArray`.  
  
Args:  

- self
- other


```Mojo
__rmul__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `Scalar * ComplexNDArray`.  
  
Args:  

- self
- other


```Mojo
__rmul__(self, other: NDArray[dtype]) -> Self
```  
Summary  
  
Enables `NDArray * ComplexNDArray`.  
  
Args:  

- self
- other

#### __rtruediv__


```Mojo
__rtruediv__(mut self, other: ComplexSIMD[cdtype, dtype=dtype]) -> Self
```  
Summary  
  
Enables `ComplexSIMD / ComplexNDArray`.  
  
Args:  

- self
- other


```Mojo
__rtruediv__(mut self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Enables `Scalar / ComplexNDArray`.  
  
Args:  

- self
- other


```Mojo
__rtruediv__(mut self, other: NDArray[dtype]) -> Self
```  
Summary  
  
Enables `NDArray / ComplexNDArray`.  
  
Args:  

- self
- other

#### __iadd__


```Mojo
__iadd__(mut self, other: ComplexSIMD[cdtype, dtype=dtype])
```  
Summary  
  
Enables `ComplexNDArray += ComplexSIMD`.  
  
Args:  

- self
- other


```Mojo
__iadd__(mut self, other: SIMD[dtype, 1])
```  
Summary  
  
Enables `ComplexNDArray += Scalar`.  
  
Args:  

- self
- other


```Mojo
__iadd__(mut self, other: Self)
```  
Summary  
  
Enables `ComplexNDArray += ComplexNDArray`.  
  
Args:  

- self
- other


```Mojo
__iadd__(mut self, other: NDArray[dtype])
```  
Summary  
  
Enables `ComplexNDArray += NDArray`.  
  
Args:  

- self
- other

#### __isub__


```Mojo
__isub__(mut self, other: ComplexSIMD[cdtype, dtype=dtype])
```  
Summary  
  
Enables `ComplexNDArray -= ComplexSIMD`.  
  
Args:  

- self
- other


```Mojo
__isub__(mut self, other: SIMD[dtype, 1])
```  
Summary  
  
Enables `ComplexNDArray -= Scalar`.  
  
Args:  

- self
- other


```Mojo
__isub__(mut self, other: Self)
```  
Summary  
  
Enables `ComplexNDArray -= ComplexNDArray`.  
  
Args:  

- self
- other


```Mojo
__isub__(mut self, other: NDArray[dtype])
```  
Summary  
  
Enables `ComplexNDArray -= NDArray`.  
  
Args:  

- self
- other

#### __imul__


```Mojo
__imul__(mut self, other: ComplexSIMD[cdtype, dtype=dtype])
```  
Summary  
  
Enables `ComplexNDArray *= ComplexSIMD`.  
  
Args:  

- self
- other


```Mojo
__imul__(mut self, other: SIMD[dtype, 1])
```  
Summary  
  
Enables `ComplexNDArray *= Scalar`.  
  
Args:  

- self
- other


```Mojo
__imul__(mut self, other: Self)
```  
Summary  
  
Enables `ComplexNDArray *= ComplexNDArray`.  
  
Args:  

- self
- other


```Mojo
__imul__(mut self, other: NDArray[dtype])
```  
Summary  
  
Enables `ComplexNDArray *= NDArray`.  
  
Args:  

- self
- other

#### __itruediv__


```Mojo
__itruediv__(mut self, other: ComplexSIMD[cdtype, dtype=dtype])
```  
Summary  
  
Enables `ComplexNDArray /= ComplexSIMD`.  
  
Args:  

- self
- other


```Mojo
__itruediv__(mut self, other: SIMD[dtype, 1])
```  
Summary  
  
Enables `ComplexNDArray /= Scalar`.  
  
Args:  

- self
- other


```Mojo
__itruediv__(mut self, other: Self)
```  
Summary  
  
Enables `ComplexNDArray /= ComplexNDArray`.  
  
Args:  

- self
- other


```Mojo
__itruediv__(mut self, other: NDArray[dtype])
```  
Summary  
  
Enables `ComplexNDArray /= NDArray`.  
  
Args:  

- self
- other

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
  
Compute the "official" string representation of ComplexNDArray. An example is: ``` fn main() raises:     var A = ComplexNDArray[cf32](List[ComplexSIMD[cf32]](14,97,-59,-4,112,), shape=List[Int](5,))     print(repr(A)) ``` It prints what can be used to construct the array itself: ```console     ComplexNDArray[cf32](List[ComplexSIMD[cf32]](14,97,-59,-4,112,), shape=List[Int](5,)) ```.  
  
Args:  

- self

#### __len__


```Mojo
__len__(self) -> Int
```  
Summary  
  
  
  
Args:  

- self

#### load


```Mojo
load[width: Int = 1](self, index: Int) -> ComplexSIMD[cdtype, dtype=dtype]
```  
Summary  
  
Safely loads a SIMD element of size `width` at `index` from the underlying buffer.  
  
Parameters:  

- width Defualt: `1`
  
Args:  

- self
- index


To bypass boundary checks, use `self._buf.ptr.load` directly.

#### store


```Mojo
store[width: Int](mut self, index: Int, val: ComplexSIMD[cdtype, dtype=dtype])
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

#### item


```Mojo
item(self, owned index: Int) -> ComplexSIMD[cdtype, dtype=dtype]
```  
Summary  
  
Return the scalar at the coordinates.  
  
Args:  

- self
- index: Index of item, counted in row-major way.


If one index is given, get the i-th item of the ComplexNDArray (not buffer).
It first scans over the first row, even it is a colume-major array.

If more than one index is given, the length of the indices must match
the number of dimensions of the array.


```Mojo
item(self, *index: Int) -> ComplexSIMD[cdtype, dtype=dtype]
```  
Summary  
  
Return the scalar at the coordinates.  
  
Args:  

- self
- \*index: The coordinates of the item.


If one index is given, get the i-th item of the ComplexNDArray (not buffer).
It first scans over the first row, even it is a colume-major array.

If more than one index is given, the length of the indices must match
the number of dimensions of the array.

#### itemset


```Mojo
itemset(mut self, index: Variant[Int, List[Int]], item: ComplexSIMD[cdtype, dtype=dtype])
```  
Summary  
  
Set the scalar at the coordinates.  
  
Args:  

- self
- index: The coordinates of the item. Can either be `Int` or `List[Int]`. If `Int` is passed, it is the index of i-th item of the whole array. If `List[Int]` is passed, it is the coordinate of the item.
- item: The scalar to be set.

#### conj


```Mojo
conj(self) -> Self
```  
Summary  
  
Return the complex conjugate of the ComplexNDArray.  
  
Args:  

- self

#### to_ndarray


```Mojo
to_ndarray(self, type: String = String("re")) -> NDArray[dtype]
```  
Summary  
  
  
  
Args:  

- self
- type Default: String("re")
