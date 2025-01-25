



# ndshape

##  Module Summary
  
Implements NDArrayShape type.
## Aliases
  
`Shape`: 
## NDArrayShape

### NDArrayShape Summary
  
  
Implements the NDArrayShape.  

### Parent Traits
  

- AnyType
- Stringable
- UnknownDestructibility
- Writable

### Fields
  
  
* size `Int`  
    - Total number of elements of corresponding array.  
* ndim `Int`  
    - Number of dimensions of array.  

### Functions

#### __init__


```Mojo
__init__(out self, shape: Int)
```  
Summary  
  
Initializes the NDArrayShape with one dimension.  
  
Args:  

- self
- shape: Size of the array.


```Mojo
__init__(out self, *shape: Int)
```  
Summary  
  
Initializes the NDArrayShape with variable shape dimensions.  
  
Args:  

- self
- \*shape: Variable number of integers representing the shape dimensions.


```Mojo
__init__(out self, *shape: Int, *, size: Int)
```  
Summary  
  
Initializes the NDArrayShape with variable shape dimensions and a specified size.  
  
Args:  

- self
- \*shape: Variable number of integers representing the shape dimensions.
- size: The total number of elements in the array.


```Mojo
__init__(out self, shape: List[Int])
```  
Summary  
  
Initializes the NDArrayShape with a list of shape dimensions.  
  
Args:  

- self
- shape: A list of integers representing the shape dimensions.


```Mojo
__init__(out self, shape: List[Int], size: Int)
```  
Summary  
  
Initializes the NDArrayShape with a list of shape dimensions and a specified size.  
  
Args:  

- self
- shape: A list of integers representing the shape dimensions.
- size: The specified size of the NDArrayShape.


```Mojo
__init__(out self, shape: VariadicList[Int])
```  
Summary  
  
Initializes the NDArrayShape with a list of shape dimensions.  
  
Args:  

- self
- shape: A list of integers representing the shape dimensions.


```Mojo
__init__(out self, shape: VariadicList[Int], size: Int)
```  
Summary  
  
Initializes the NDArrayShape with a list of shape dimensions and a specified size.  
  
Args:  

- self
- shape: A list of integers representing the shape dimensions.
- size: The specified size of the NDArrayShape.


```Mojo
__init__(out self, shape: Self)
```  
Summary  
  
Initializes the NDArrayShape with another NDArrayShape.  
  
Args:  

- self
- shape: Another NDArrayShape to initialize from.

#### __getitem__


```Mojo
__getitem__(self, index: Int) -> Int
```  
Summary  
  
Get shape at specified index.  
  
Args:  

- self
- index

#### __setitem__


```Mojo
__setitem__(mut self, index: Int, val: Int)
```  
Summary  
  
Set shape at specified index.  
  
Args:  

- self
- index
- val

#### __eq__


```Mojo
__eq__(self, other: Self) -> Bool
```  
Summary  
  
Check if two arrayshapes have identical dimensions.  
  
Args:  

- self
- other

#### __ne__


```Mojo
__ne__(self, other: Self) -> Bool
```  
Summary  
  
Check if two arrayshapes don't have identical dimensions.  
  
Args:  

- self
- other

#### __contains__


```Mojo
__contains__(self, val: Int) -> Bool
```  
Summary  
  
Check if any of the dimensions are equal to a value.  
  
Args:  

- self
- val

#### __len__


```Mojo
__len__(self) -> Int
```  
Summary  
  
Get number of dimensions of the array described by arrayshape.  
  
Args:  

- self

#### __str__


```Mojo
__str__(self) -> String
```  
Summary  
  
Return a string of the shape of the array described by arrayshape.  
  
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
