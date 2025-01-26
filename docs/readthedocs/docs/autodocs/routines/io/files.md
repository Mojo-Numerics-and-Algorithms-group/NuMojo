



# files

##  Module Summary
  

## loadtxt


```Mojo
loadtxt[dtype: DType = float64](filename: String, delimiter: String = String(","), skiprows: Int = 0, usecols: Optional[List[Int]] = Optional(None)) -> NDArray[dtype]
```  
Summary  
  
  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- filename
- delimiter Default: String(",")
- skiprows Default: 0
- usecols Default: Optional(None)

## savetxt


```Mojo
savetxt[dtype: DType = float64](filename: String, array: NDArray[dtype], delimiter: String = String(","))
```  
Summary  
  
  
  
Parameters:  

- dtype Defualt: `float64`
  
Args:  

- filename
- array
- delimiter Default: String(",")
