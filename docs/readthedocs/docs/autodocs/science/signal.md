



# signal

##  Module Summary
  
Implements signal processing.
## convolve2d


```Mojo
convolve2d[dtype: DType, //](in1: NDArray[dtype], in2: NDArray[dtype]) -> NDArray[dtype]
```  
Summary  
  
Convolve two 2-dimensional arrays.  
  
Parameters:  

- dtype
  
Args:  

- in1: Input array 1.
- in2: Input array 2. It should be of a smaller size of in1.


Currently, the mode is "valid".

TODO: Add more modes.

Example:
```mojo
import numojo as nm
fn main() raises:
    var in1 = nm.routines.creation.fromstring("[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]")
    var in2 = nm.routines.creation.fromstring("[[1, 0], [0, -1]]")
    print(nm.science.signal.convolve2d(in1, in2))
```