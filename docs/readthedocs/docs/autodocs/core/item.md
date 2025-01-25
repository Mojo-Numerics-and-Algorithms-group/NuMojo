



# item

##  Module Summary
  
Implements Item type.
## Aliases
  
`item`: 
## Item

### Item Summary
  
  
  

### Parent Traits
  

- AnyType
- CollectionElement
- Copyable
- Movable
- Stringable
- UnknownDestructibility

### Fields
  
  
* ndim `Int`  

### Functions

#### __init__


```Mojo
__init__[T: Indexer](out self, *args: T)
```  
Summary  
  
Construct the tuple.  
  
Parameters:  

- T
  
Args:  

- self
- \*args: Initial values.


Parameter:
    T: Type of values. It can be converted to `Int` with `index()`.


```Mojo
__init__[T: IndexerCollectionElement](out self, args: List[T])
```  
Summary  
  
Construct the tuple.  
  
Parameters:  

- T
  
Args:  

- self
- args: Initial values.


Parameter:
    T: Type of values. It can be converted to `Int` with `index()`.


```Mojo
__init__(out self, args: VariadicList[Int])
```  
Summary  
  
Construct the tuple.  
  
Args:  

- self
- args: Initial values.


```Mojo
__init__(out self, ndim: Int, initialized: Bool)
```  
Summary  
  
Construct Item with number of dimensions.  
  
Args:  

- self
- ndim: Number of dimensions.
- initialized: Whether the shape is initialized. If yes, the values will be set to 0. If no, the values will be uninitialized.


This method is useful when you want to create a Item with given ndim
without knowing the Item values.

#### __copyinit__


```Mojo
__copyinit__(out self, other: Self)
```  
Summary  
  
Copy construct the tuple.  
  
Args:  

- self
- other: The tuple to copy.

#### __del__


```Mojo
__del__(owned self)
```  
Summary  
  
  
  
Args:  

- self

#### __getitem__


```Mojo
__getitem__[T: Indexer](self, idx: T) -> Int
```  
Summary  
  
Get the value at the specified index.  
  
Parameters:  

- T
  
Args:  

- self
- idx: The index of the value to get.


Parameter:
    T: Type of values. It can be converted to `Int` with `index()`.

#### __setitem__


```Mojo
__setitem__[T: Indexer, U: Indexer](self, idx: T, val: U)
```  
Summary  
  
Set the value at the specified index.  
  
Parameters:  

- T
- U
  
Args:  

- self
- idx: The index of the value to set.
- val: The value to set.


Parameter:
    T: Type of values. It can be converted to `Int` with `index()`.
    U: Type of values. It can be converted to `Int` with `index()`.

#### __len__


```Mojo
__len__(self) -> Int
```  
Summary  
  
Get the length of the tuple.  
  
Args:  

- self

#### __iter__


```Mojo
__iter__(self) -> _ItemIter
```  
Summary  
  
Iterate over elements of the NDArray, returning copied value.  
  
Args:  

- self


Notes:
    Need to add lifetimes after the new release.
#### __repr__


```Mojo
__repr__(self) -> String
```  
Summary  
  
  
  
Args:  

- self

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
