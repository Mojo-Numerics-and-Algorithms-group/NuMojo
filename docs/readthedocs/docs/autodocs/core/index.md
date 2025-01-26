



# index

##  Module Summary
  
Implements Idx type.
## Idx

### Idx Summary
  
  
  

### Parent Traits
  

- AnyType
- CollectionElement
- Copyable
- Movable
- UnknownDestructibility

### Aliases
  
`dtype`:   
`width`: 
### Fields
  
  
* storage `UnsafePointer[SIMD[index, 1]]`  
* len `Int`  

### Functions

#### __init__


```Mojo
__init__(out self, owned *args: SIMD[index, 1])
```  
Summary  
  
Construct the tuple.  
  
Args:  

- self
- \*args: Initial values.


```Mojo
__init__(out self, owned *args: Int)
```  
Summary  
  
Construct the tuple.  
  
Args:  

- self
- \*args: Initial values.


```Mojo
__init__(out self, owned args: Variant[List[Int], VariadicList[Int]])
```  
Summary  
  
Construct the tuple.  
  
Args:  

- self
- args: Initial values.

#### __copyinit__


```Mojo
__copyinit__(out self, other: Self)
```  
Summary  
  
Copy construct the tuple.  
  
Args:  

- self
- other: The tuple to copy.

#### __moveinit__


```Mojo
__moveinit__(out self, owned other: Self)
```  
Summary  
  
Move construct the tuple.  
  
Args:  

- self
- other: The tuple to move.

#### __getitem__


```Mojo
__getitem__(self, index: Int) -> Int
```  
Summary  
  
Get the value at the specified index.  
  
Args:  

- self
- index: The index of the value to get.

#### __setitem__


```Mojo
__setitem__(self, index: Int, val: Int)
```  
Summary  
  
Set the value at the specified index.  
  
Args:  

- self
- index: The index of the value to set.
- val: The value to set.

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
__iter__(self) -> _IdxIter[self]
```  
Summary  
  
Iterate over elements of the NDArray, returning copied value.  
  
Args:  

- self


Notes:
    Need to add lifetimes after the new release.
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

#### str


```Mojo
str(self) -> String
```  
Summary  
  
  
  
Args:  

- self

#### load


```Mojo
load[width: Int = 1](self, index: Int) -> SIMD[index, width]
```  
Summary  
  
  
  
Parameters:  

- width Defualt: `1`
  
Args:  

- self
- index

#### store


```Mojo
store[width: Int = 1](mut self, index: Int, val: SIMD[index, width])
```  
Summary  
  
  
  
Parameters:  

- width Defualt: `1`
  
Args:  

- self
- index
- val

#### load_unsafe


```Mojo
load_unsafe[width: Int = 1](self, index: Int) -> SIMD[index, width]
```  
Summary  
  
  
  
Parameters:  

- width Defualt: `1`
  
Args:  

- self
- index

#### store_unsafe


```Mojo
store_unsafe[width: Int = 1](mut self, index: Int, val: SIMD[index, width])
```  
Summary  
  
  
  
Parameters:  

- width Defualt: `1`
  
Args:  

- self
- index
- val
