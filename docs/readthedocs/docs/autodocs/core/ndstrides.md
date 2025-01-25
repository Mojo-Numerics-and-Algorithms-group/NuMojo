



# ndstrides

##  Module Summary
  
Implements NDArrayStrides type.
## Aliases
  
`Strides`: 
## NDArrayStrides

### NDArrayStrides Summary
  
  
Implements the NDArrayStrides.  

### Parent Traits
  

- AnyType
- Stringable
- UnknownDestructibility

### Fields
  
  
* ndim `Int`  
    - Number of dimensions of array.  

### Functions

#### __init__


```Mojo
__init__(out self, *strides: Int)
```  
Summary  
  
  
  
Args:  

- self
- \*strides


```Mojo
__init__(out self, strides: List[Int])
```  
Summary  
  
  
  
Args:  

- self
- strides


```Mojo
__init__(out self, strides: VariadicList[Int])
```  
Summary  
  
  
  
Args:  

- self
- strides


```Mojo
__init__(out self, strides: Self)
```  
Summary  
  
  
  
Args:  

- self
- strides


```Mojo
__init__(out self, *shape: Int, *, order: String)
```  
Summary  
  
  
  
Args:  

- self
- \*shape
- order


```Mojo
__init__(out self, shape: List[Int], order: String = String("C"))
```  
Summary  
  
  
  
Args:  

- self
- shape
- order Default: String("C")


```Mojo
__init__(out self, shape: VariadicList[Int], order: String = String("C"))
```  
Summary  
  
  
  
Args:  

- self
- shape
- order Default: String("C")


```Mojo
__init__(out self, owned shape: NDArrayShape, order: String = String("C"))
```  
Summary  
  
  
  
Args:  

- self
- shape
- order Default: String("C")

#### __copyinit__


```Mojo
__copyinit__(out self, other: Self)
```  
Summary  
  
  
  
Args:  

- self
- other

#### __getitem__


```Mojo
__getitem__(self, index: Int) -> Int
```  
Summary  
  
  
  
Args:  

- self
- index

#### __setitem__


```Mojo
__setitem__(mut self, index: Int, val: Int)
```  
Summary  
  
  
  
Args:  

- self
- index
- val

#### __eq__


```Mojo
__eq__(self, other: Self) -> Bool
```  
Summary  
  
  
  
Args:  

- self
- other

#### __ne__


```Mojo
__ne__(self, other: Self) -> Bool
```  
Summary  
  
  
  
Args:  

- self
- other

#### __contains__


```Mojo
__contains__(self, val: Int) -> Bool
```  
Summary  
  
  
  
Args:  

- self
- val

#### __len__


```Mojo
__len__(self) -> Int
```  
Summary  
  
  
  
Args:  

- self

#### __repr__


```Mojo
__repr__(self) -> String
```  
Summary  
  
Return a string of the strides of the array.  
  
Args:  

- self

#### __str__


```Mojo
__str__(self) -> String
```  
Summary  
  
Return a string of the strides of the array.  
  
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
