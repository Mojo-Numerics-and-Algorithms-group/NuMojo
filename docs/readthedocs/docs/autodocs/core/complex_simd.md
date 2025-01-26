



# complex_simd

##  Module Summary
  

## Aliases
  
`ComplexScalar`: 
## ComplexSIMD

### ComplexSIMD Summary
  
  
Represents a SIMD[dtype, 1] Complex number with real and imaginary parts.  

### Parent Traits
  

- AnyType
- Stringable
- UnknownDestructibility
- Writable

### Fields
  
  
* re `SIMD[dtype, size]`  
* im `SIMD[dtype, size]`  

### Functions

#### __init__


```Mojo
__init__(out self, other: Self)
```  
Summary  
  
Initializes a ComplexSIMD instance by copying another instance.  
  
Args:  

- self
- other


Arguments:
    other: Another ComplexSIMD instance to copy from.

```Mojo
__init__(out self, re: SIMD[dtype, size], im: SIMD[dtype, size])
```  
Summary  
  
Initializes a ComplexSIMD instance with specified real and imaginary parts.  
  
Args:  

- self
- re
- im


Arguments:
    re: The real part of the complex number.
    im: The imaginary part of the complex number.

Example:
```mojo
var A = ComplexSIMD[cf32](SIMD[f32, 1](1.0), SIMD[f32, 1](2.0))
var B = ComplexSIMD[cf32](SIMD[f32, 1](3.0), SIMD[f32, 1](4.0))
var C = A + B
print(C)
```

```Mojo
__init__(out self, val: SIMD[dtype, size])
```  
Summary  
  
Initializes a ComplexSIMD instance with specified real and imaginary parts.  
  
Args:  

- self
- val


Arguments:
    re: The real part of the complex number.
    im: The imaginary part of the complex number.
#### __getitem__


```Mojo
__getitem__(self, idx: Int) -> SIMD[dtype, size]
```  
Summary  
  
Gets the real or imaginary part of the ComplexSIMD instance.  
  
Args:  

- self
- idx


Arguments:
    self: The ComplexSIMD instance.
    idx: The index to access (0 for real, 1 for imaginary).

#### __setitem__


```Mojo
__setitem__(mut self, idx: Int, value: Self)
```  
Summary  
  
Sets the real and imaginary parts of the ComplexSIMD instance.  
  
Args:  

- self
- idx
- value


Arguments:
    self: The ComplexSIMD instance to modify.
    idx: The index to access (0 for real, 1 for imaginary).
    value: The new value to set.

```Mojo
__setitem__(mut self, idx: Int, re: SIMD[dtype, size], im: SIMD[dtype, size])
```  
Summary  
  
Sets the real and imaginary parts of the ComplexSIMD instance.  
  
Args:  

- self
- idx
- re
- im


Arguments:
    self: The ComplexSIMD instance to modify.
    idx: The index to access (0 for real, 1 for imaginary).
    re: The new value for the real part.
    im: The new value for the imaginary part.
#### __neg__


```Mojo
__neg__(self) -> Self
```  
Summary  
  
Negates the ComplexSIMD instance.  
  
Args:  

- self

#### __pos__


```Mojo
__pos__(self) -> Self
```  
Summary  
  
Returns the ComplexSIMD instance itself.  
  
Args:  

- self

#### __eq__


```Mojo
__eq__(self, other: Self) -> Bool
```  
Summary  
  
Checks if two ComplexSIMD instances are equal.  
  
Args:  

- self
- other


Arguments:
    self: The first ComplexSIMD instance.
    other: The second ComplexSIMD instance to compare with.

#### __ne__


```Mojo
__ne__(self, other: Self) -> Bool
```  
Summary  
  
Checks if two ComplexSIMD instances are not equal.  
  
Args:  

- self
- other


Arguments:
    self: The first ComplexSIMD instance.
    other: The second ComplexSIMD instance to compare with.

#### __add__


```Mojo
__add__(self, other: Self) -> Self
```  
Summary  
  
Adds two ComplexSIMD instances.  
  
Args:  

- self
- other


Arguments:
    other: The ComplexSIMD instance to add.

#### __sub__


```Mojo
__sub__(self, other: Self) -> Self
```  
Summary  
  
Subtracts another ComplexSIMD instance from this one.  
  
Args:  

- self
- other


Arguments:
    other: The ComplexSIMD instance to subtract.

#### __mul__


```Mojo
__mul__(self, other: Self) -> Self
```  
Summary  
  
Multiplies two ComplexSIMD instances.  
  
Args:  

- self
- other


Arguments:
    other: The ComplexSIMD instance to multiply with.

#### __truediv__


```Mojo
__truediv__(self, other: Self) -> Self
```  
Summary  
  
Divides this ComplexSIMD instance by another.  
  
Args:  

- self
- other


Arguments:
    other: The ComplexSIMD instance to divide by.

#### __iadd__


```Mojo
__iadd__(mut self, other: Self)
```  
Summary  
  
Performs in-place addition of another ComplexSIMD instance.  
  
Args:  

- self
- other


Arguments:
    other: The ComplexSIMD instance to add.
#### __isub__


```Mojo
__isub__(mut self, other: Self)
```  
Summary  
  
Performs in-place subtraction of another ComplexSIMD instance.  
  
Args:  

- self
- other


Arguments:
    other: The ComplexSIMD instance to subtract.
#### __imul__


```Mojo
__imul__(mut self, other: Self)
```  
Summary  
  
Performs in-place multiplication with another ComplexSIMD instance.  
  
Args:  

- self
- other


Arguments:
    other: The ComplexSIMD instance to multiply with.
#### __itruediv__


```Mojo
__itruediv__(mut self, other: Self)
```  
Summary  
  
Performs in-place division by another ComplexSIMD instance.  
  
Args:  

- self
- other


Arguments:
    other: The ComplexSIMD instance to divide by.
#### __str__


```Mojo
__str__(self) -> String
```  
Summary  
  
Returns a string representation of the ComplexSIMD instance.  
  
Args:  

- self

#### write_to


```Mojo
write_to[W: Writer](self, mut writer: W)
```  
Summary  
  
Writes the ComplexSIMD instance to a writer.  
  
Parameters:  

- W
  
Args:  

- self
- writer


Arguments:
    self: The ComplexSIMD instance to write.
    writer: The writer to write to.
#### __repr__


```Mojo
__repr__(self) -> String
```  
Summary  
  
Returns a string representation of the ComplexSIMD instance.  
  
Args:  

- self

#### __abs__


```Mojo
__abs__(self) -> SIMD[dtype, size]
```  
Summary  
  
Returns the magnitude of the ComplexSIMD instance.  
  
Args:  

- self

#### norm


```Mojo
norm(self) -> SIMD[dtype, size]
```  
Summary  
  
Returns the squared magnitude of the ComplexSIMD instance.  
  
Args:  

- self

#### conj


```Mojo
conj(self) -> Self
```  
Summary  
  
Returns the complex conjugate of the ComplexSIMD instance.  
  
Args:  

- self

#### real


```Mojo
real(self) -> SIMD[dtype, size]
```  
Summary  
  
Returns the real part of the ComplexSIMD instance.  
  
Args:  

- self

#### imag


```Mojo
imag(self) -> SIMD[dtype, size]
```  
Summary  
  
Returns the imaginary part of the ComplexSIMD instance.  
  
Args:  

- self
