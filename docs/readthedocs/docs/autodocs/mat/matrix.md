



# matrix

##  Module Summary
  
`numojo.mat.matrix` provides:
## Matrix

### Matrix Summary
  
  
`Matrix` is a special case of `NDArray` (2DArray) but has some targeted optimization since the number of dimensions is known at the compile time. It gains some advantages in running speed, which is very useful when users only want to work with 2-dimensional arrays. The indexing and slicing is also more consistent with `numpy`.  

### Parent Traits
  

- AnyType
- CollectionElement
- Copyable
- Movable
- Stringable
- UnknownDestructibility
- Writable

### Aliases
  
`width`: Vector size of the data type.
### Fields
  
  
* shape `Tuple[Int, Int]`  
    - Shape of Matrix.  
* size `Int`  
    - Size of Matrix.  
* strides `Tuple[Int, Int]`  
    - Strides of matrix.  

### Functions

#### __init__


```Mojo
__init__(out self, shape: Tuple[Int, Int])
```  
Summary  
  
Matrix NDArray initialization.  
  
Args:  

- self
- shape: List of shape.


```Mojo
__init__(out self, data: Self)
```  
Summary  
  
Create a matrix from a matrix.  
  
Args:  

- self
- data


```Mojo
__init__(out self, data: NDArray[dtype])
```  
Summary  
  
Create Matrix from NDArray.  
  
Args:  

- self
- data

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
__moveinit__(out self, owned other: Self)
```  
Summary  
  
Move other into self.  
  
Args:  

- self
- other

#### __del__


```Mojo
__del__(owned self)
```  
Summary  
  
  
  
Args:  

- self

#### __getitem__


```Mojo
__getitem__(self, owned x: Int, owned y: Int) -> SIMD[dtype, 1]
```  
Summary  
  
Return the scalar at the index.  
  
Args:  

- self
- x: The row number.
- y: The column number.


```Mojo
__getitem__(self, owned x: Int) -> Self
```  
Summary  
  
Return the corresponding row at the index.  
  
Args:  

- self
- x: The row number.


```Mojo
__getitem__(self, x: Slice, y: Slice) -> Self
```  
Summary  
  
Get item from two slices.  
  
Args:  

- self
- x
- y


```Mojo
__getitem__(self, x: Slice, owned y: Int) -> Self
```  
Summary  
  
Get item from one slice and one int.  
  
Args:  

- self
- x
- y


```Mojo
__getitem__(self, owned x: Int, y: Slice) -> Self
```  
Summary  
  
Get item from one int and one slice.  
  
Args:  

- self
- x
- y


```Mojo
__getitem__(self, indices: List[Int]) -> Self
```  
Summary  
  
Get item by a list of integers.  
  
Args:  

- self
- indices

#### __setitem__


```Mojo
__setitem__(self, x: Int, y: Int, value: SIMD[dtype, 1])
```  
Summary  
  
Return the scalar at the index.  
  
Args:  

- self
- x: The row number.
- y: The column number.
- value: The value to be set.


```Mojo
__setitem__(self, owned x: Int, value: Self)
```  
Summary  
  
Set the corresponding row at the index with the given matrix.  
  
Args:  

- self
- x: The row number.
- value: Matrix (row vector).

#### __lt__


```Mojo
__lt__(self, other: Self) -> Matrix[bool]
```  
Summary  
  
  
  
Args:  

- self
- other


```Mojo
__lt__(self, other: SIMD[dtype, 1]) -> Matrix[bool]
```  
Summary  
  
Matrix less than scalar.  
  
Args:  

- self
- other


```mojo
from numojo.mat import ones
A = ones(shape=(4, 4))
print(A < 2)
```
#### __le__


```Mojo
__le__(self, other: Self) -> Matrix[bool]
```  
Summary  
  
  
  
Args:  

- self
- other


```Mojo
__le__(self, other: SIMD[dtype, 1]) -> Matrix[bool]
```  
Summary  
  
Matrix less than and equal to scalar.  
  
Args:  

- self
- other


```mojo
from numojo.mat import ones
A = ones(shape=(4, 4))
print(A <= 2)
```
#### __eq__


```Mojo
__eq__(self, other: Self) -> Matrix[bool]
```  
Summary  
  
  
  
Args:  

- self
- other


```Mojo
__eq__(self, other: SIMD[dtype, 1]) -> Matrix[bool]
```  
Summary  
  
Matrix less than and equal to scalar.  
  
Args:  

- self
- other


```mojo
from numojo.mat import ones
A = ones(shape=(4, 4))
print(A == 2)
```
#### __ne__


```Mojo
__ne__(self, other: Self) -> Matrix[bool]
```  
Summary  
  
  
  
Args:  

- self
- other


```Mojo
__ne__(self, other: SIMD[dtype, 1]) -> Matrix[bool]
```  
Summary  
  
Matrix less than and equal to scalar.  
  
Args:  

- self
- other


```mojo
from numojo.mat import ones
A = ones(shape=(4, 4))
print(A != 2)
```
#### __gt__


```Mojo
__gt__(self, other: Self) -> Matrix[bool]
```  
Summary  
  
  
  
Args:  

- self
- other


```Mojo
__gt__(self, other: SIMD[dtype, 1]) -> Matrix[bool]
```  
Summary  
  
Matrix greater than scalar.  
  
Args:  

- self
- other


```mojo
from numojo.mat import ones
A = ones(shape=(4, 4))
print(A > 2)
```
#### __ge__


```Mojo
__ge__(self, other: Self) -> Matrix[bool]
```  
Summary  
  
  
  
Args:  

- self
- other


```Mojo
__ge__(self, other: SIMD[dtype, 1]) -> Matrix[bool]
```  
Summary  
  
Matrix greater than and equal to scalar.  
  
Args:  

- self
- other


```mojo
from numojo.mat import ones
A = ones(shape=(4, 4))
print(A >= 2)
```
#### __add__


```Mojo
__add__(self, other: Self) -> Self
```  
Summary  
  
  
  
Args:  

- self
- other


```Mojo
__add__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Add matrix to scalar.  
  
Args:  

- self
- other


```mojo
from numojo.mat import ones
A = ones(shape=(4, 4))
print(A + 2)
```
#### __sub__


```Mojo
__sub__(self, other: Self) -> Self
```  
Summary  
  
  
  
Args:  

- self
- other


```Mojo
__sub__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Substract matrix by scalar.  
  
Args:  

- self
- other


```mojo
from numojo.mat import ones
A = ones(shape=(4, 4))
print(A - 2)
```
#### __mul__


```Mojo
__mul__(self, other: Self) -> Self
```  
Summary  
  
  
  
Args:  

- self
- other


```Mojo
__mul__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Mutiply matrix by scalar.  
  
Args:  

- self
- other


```mojo
from numojo.mat import ones
A = ones(shape=(4, 4))
print(A * 2)
```
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
__truediv__(self, other: Self) -> Self
```  
Summary  
  
  
  
Args:  

- self
- other


```Mojo
__truediv__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Divide matrix by scalar.  
  
Args:  

- self
- other

#### __pow__


```Mojo
__pow__(self, rhs: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Power of items.  
  
Args:  

- self
- rhs

#### __radd__


```Mojo
__radd__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Right-add.  
  
Args:  

- self
- other


```mojo
from numojo.mat import ones
A = ones(shape=(4, 4))
print(2 + A)
```
#### __rsub__


```Mojo
__rsub__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Right-sub.  
  
Args:  

- self
- other


```mojo
from numojo.mat import ones
A = ones(shape=(4, 4))
print(2 - A)
```
#### __rmul__


```Mojo
__rmul__(self, other: SIMD[dtype, 1]) -> Self
```  
Summary  
  
Right-mul.  
  
Args:  

- self
- other


```mojo
from numojo.mat import ones
A = ones(shape=(4, 4))
print(2 * A)
```
#### __str__


```Mojo
__str__(self) -> String
```  
Summary  
  
  
  
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

#### __iter__


```Mojo
__iter__(self) -> _MatrixIter[self, dtype]
```  
Summary  
  
Iterate over elements of the Matrix, returning copied value.  
  
Args:  

- self


Example:
```mojo
from numojo import mat
var A = mat.rand((4,4))
for i in A:
    print(i)
```

#### __reversed__


```Mojo
__reversed__(self) -> _MatrixIter[self, dtype, False]
```  
Summary  
  
Iterate backwards over elements of the Matrix, returning copied value.  
  
Args:  

- self

#### all


```Mojo
all(self) -> SIMD[dtype, 1]
```  
Summary  
  
Test whether all array elements evaluate to True.  
  
Args:  

- self


```Mojo
all(self, axis: Int) -> Self
```  
Summary  
  
Test whether all array elements evaluate to True along axis.  
  
Args:  

- self
- axis

#### any


```Mojo
any(self) -> SIMD[dtype, 1]
```  
Summary  
  
Test whether any array elements evaluate to True.  
  
Args:  

- self


```Mojo
any(self, axis: Int) -> Self
```  
Summary  
  
Test whether any array elements evaluate to True along axis.  
  
Args:  

- self
- axis

#### argmax


```Mojo
argmax(self) -> SIMD[index, 1]
```  
Summary  
  
Index of the max. It is first flattened before sorting.  
  
Args:  

- self


```Mojo
argmax(self, axis: Int) -> Matrix[index]
```  
Summary  
  
Index of the max along the given axis.  
  
Args:  

- self
- axis

#### argmin


```Mojo
argmin(self) -> SIMD[index, 1]
```  
Summary  
  
Index of the min. It is first flattened before sorting.  
  
Args:  

- self


```Mojo
argmin(self, axis: Int) -> Matrix[index]
```  
Summary  
  
Index of the min along the given axis.  
  
Args:  

- self
- axis

#### argsort


```Mojo
argsort(self) -> Matrix[index]
```  
Summary  
  
Argsort the Matrix. It is first flattened before sorting.  
  
Args:  

- self


```Mojo
argsort(self, axis: Int) -> Matrix[index]
```  
Summary  
  
Argsort the Matrix along the given axis.  
  
Args:  

- self
- axis

#### astype


```Mojo
astype[asdtype: DType](self) -> Matrix[asdtype]
```  
Summary  
  
Copy of the matrix, cast to a specified type.  
  
Parameters:  

- asdtype
  
Args:  

- self

#### cumprod


```Mojo
cumprod(self) -> Self
```  
Summary  
  
Cumprod of flattened matrix.  
  
Args:  

- self


Example:
```mojo
from numojo import mat
var A = mat.rand(shape=(100, 100))
print(A.cumprod())
```

```Mojo
cumprod(self, axis: Int) -> Self
```  
Summary  
  
Cumprod of Matrix along the axis.  
  
Args:  

- self
- axis: 0 or 1.


Example:
```mojo
from numojo import mat
var A = mat.rand(shape=(100, 100))
print(A.cumprod(axis=0))
print(A.cumprod(axis=1))
```
#### cumsum


```Mojo
cumsum(self) -> Self
```  
Summary  
  
  
  
Args:  

- self


```Mojo
cumsum(self, axis: Int) -> Self
```  
Summary  
  
  
  
Args:  

- self
- axis

#### fill


```Mojo
fill(self, fill_value: SIMD[dtype, 1])
```  
Summary  
  
Fill the matrix with value.  
  
Args:  

- self
- fill_value


See also function `mat.creation.full`.
#### flatten


```Mojo
flatten(self) -> Self
```  
Summary  
  
Return a flattened copy of the matrix.  
  
Args:  

- self

#### inv


```Mojo
inv(self) -> Self
```  
Summary  
  
Inverse of matrix.  
  
Args:  

- self

#### max


```Mojo
max(self) -> SIMD[dtype, 1]
```  
Summary  
  
Find max item. It is first flattened before sorting.  
  
Args:  

- self


```Mojo
max(self, axis: Int) -> Self
```  
Summary  
  
Find max item along the given axis.  
  
Args:  

- self
- axis

#### mean


```Mojo
mean(self) -> SIMD[dtype, 1]
```  
Summary  
  
Calculate the arithmetic average of all items in the Matrix.  
  
Args:  

- self


```Mojo
mean(self, axis: Int) -> Self
```  
Summary  
  
Calculate the arithmetic average of a Matrix along the axis.  
  
Args:  

- self
- axis: 0 or 1.

#### min


```Mojo
min(self) -> SIMD[dtype, 1]
```  
Summary  
  
Find min item. It is first flattened before sorting.  
  
Args:  

- self


```Mojo
min(self, axis: Int) -> Self
```  
Summary  
  
Find min item along the given axis.  
  
Args:  

- self
- axis

#### prod


```Mojo
prod(self) -> SIMD[dtype, 1]
```  
Summary  
  
Product of all items in the Matrix.  
  
Args:  

- self


```Mojo
prod(self, axis: Int) -> Self
```  
Summary  
  
Product of items in a Matrix along the axis.  
  
Args:  

- self
- axis: 0 or 1.


Example:
```mojo
from numojo import mat
var A = mat.rand(shape=(100, 100))
print(A.prod(axis=0))
print(A.prod(axis=1))
```
#### reshape


```Mojo
reshape(self, shape: Tuple[Int, Int]) -> Self
```  
Summary  
  
Change shape and size of matrix and return a new matrix.  
  
Args:  

- self
- shape

#### resize


```Mojo
resize(mut self, shape: Tuple[Int, Int])
```  
Summary  
  
Change shape and size of matrix in-place.  
  
Args:  

- self
- shape

#### round


```Mojo
round(self, decimals: Int) -> Self
```  
Summary  
  
  
  
Args:  

- self
- decimals

#### std


```Mojo
std(self, ddof: Int = 0) -> SIMD[dtype, 1]
```  
Summary  
  
Compute the standard deviation.  
  
Args:  

- self
- ddof: Delta degree of freedom. Default: 0


```Mojo
std(self, axis: Int, ddof: Int = 0) -> Self
```  
Summary  
  
Compute the standard deviation along axis.  
  
Args:  

- self
- axis: 0 or 1.
- ddof: Delta degree of freedom. Default: 0

#### sum


```Mojo
sum(self) -> SIMD[dtype, 1]
```  
Summary  
  
Sum up all items in the Matrix.  
  
Args:  

- self


Example:
```mojo
from numojo import mat
var A = mat.rand(shape=(100, 100))
print(A.sum())
```

```Mojo
sum(self, axis: Int) -> Self
```  
Summary  
  
Sum up the items in a Matrix along the axis.  
  
Args:  

- self
- axis: 0 or 1.


Example:
```mojo
from numojo import mat
var A = mat.rand(shape=(100, 100))
print(A.sum(axis=0))
print(A.sum(axis=1))
```
#### trace


```Mojo
trace(self) -> SIMD[dtype, 1]
```  
Summary  
  
Transpose of matrix.  
  
Args:  

- self

#### transpose


```Mojo
transpose(self) -> Self
```  
Summary  
  
Transpose of matrix.  
  
Args:  

- self

#### T


```Mojo
T(self) -> Self
```  
Summary  
  
  
  
Args:  

- self

#### variance


```Mojo
variance(self, ddof: Int = 0) -> SIMD[dtype, 1]
```  
Summary  
  
Compute the variance.  
  
Args:  

- self
- ddof: Delta degree of freedom. Default: 0


```Mojo
variance(self, axis: Int, ddof: Int = 0) -> Self
```  
Summary  
  
Compute the variance along axis.  
  
Args:  

- self
- axis: 0 or 1.
- ddof: Delta degree of freedom. Default: 0

#### to_ndarray


```Mojo
to_ndarray(self) -> NDArray[dtype]
```  
Summary  
  
Create `NDArray` from `Matrix`.  
  
Args:  

- self


It makes a copy of the buffer of the matrix.
#### to_numpy


```Mojo
to_numpy(self) -> PythonObject
```  
Summary  
  
See `numojo.core.utility.to_numpy`.  
  
Args:  

- self

## broadcast_to


```Mojo
broadcast_to[dtype: DType](A: Matrix[dtype], shape: Tuple[Int, Int]) -> Matrix[dtype]
```  
Summary  
  
Broadcase the vector to the given shape.  
  
Parameters:  

- dtype
  
Args:  

- A
- shape


Example:

```console
> from numojo import mat
> a = mat.fromstring("1 2 3", shape=(1, 3))
> print(mat.broadcast_to(a, (3, 3)))
[[1.0   2.0     3.0]
 [1.0   2.0     3.0]
 [1.0   2.0     3.0]]
> a = mat.fromstring("1 2 3", shape=(3, 1))
> print(mat.broadcast_to(a, (3, 3)))
[[1.0   1.0     1.0]
 [2.0   2.0     2.0]
 [3.0   3.0     3.0]]
> a = mat.fromstring("1", shape=(1, 1))
> print(mat.broadcast_to(a, (3, 3)))
[[1.0   1.0     1.0]
 [1.0   1.0     1.0]
 [1.0   1.0     1.0]]
> a = mat.fromstring("1 2", shape=(1, 2))
> print(mat.broadcast_to(a, (1, 2)))
[[1.0   2.0]]
> a = mat.fromstring("1 2 3 4", shape=(2, 2))
> print(mat.broadcast_to(a, (4, 2)))
Unhandled exception caught during execution: Cannot broadcast shape 2x2 to shape 4x2!
```

```Mojo
broadcast_to[dtype: DType](A: SIMD[dtype, 1], shape: Tuple[Int, Int]) -> Matrix[dtype]
```  
Summary  
  
Broadcase the scalar to the given shape.  
  
Parameters:  

- dtype
  
Args:  

- A
- shape
