



# random

##  Module Summary
  
Random values array generation.
## rand


```Mojo
rand[dtype: DType = float64](*shape: Int) -> NDArray[dtype]
```  
Summary  
  
Generate a random NDArray of the given shape and dtype.  
  
Parameters:  

- dtype: The data type of the NDArray elements. Defualt: `float64`
  
Args:  

- \*shape: The shape of the NDArray.


Example:
    ```py
    var arr = numojo.core.random.rand[numojo.i16](3,2,4)
    print(arr)
    ```


```Mojo
rand[dtype: DType = float64](*shape: Int, *, min: SIMD[dtype, 1], max: SIMD[dtype, 1]) -> NDArray[dtype]
```  
Summary  
  
Generate a random NDArray of the given shape and dtype with values between `min` and `max`.  
  
Parameters:  

- dtype: The data type of the NDArray elements. Defualt: `float64`
  
Args:  

- \*shape: The shape of the NDArray.
- min: The minimum value of the random values.
- max: The maximum value of the random values.


Example:
    ```py
    var arr = numojo.core.random.rand[numojo.i16](3,2,4, min=0, max=100)
    print(arr)
    ```

```Mojo
rand[dtype: DType = float64](shape: List[Int], min: SIMD[dtype, 1], max: SIMD[dtype, 1]) -> NDArray[dtype]
```  
Summary  
  
Generate a random NDArray of the given shape and dtype with values between `min` and `max`.  
  
Parameters:  

- dtype: The data type of the NDArray elements. Defualt: `float64`
  
Args:  

- shape: The shape of the NDArray.
- min: The minimum value of the random values.
- max: The maximum value of the random values.


Example:
    ```py
    var arr = numojo.core.random.rand[numojo.i16]((3,2,4), min=0, max=100)
    print(arr)
    ```

## randn


```Mojo
randn[dtype: DType = float64](*shape: Int, *, mean: SIMD[dtype, 1] = SIMD(0), variance: SIMD[dtype, 1] = SIMD(1)) -> NDArray[dtype]
```  
Summary  
  
Generate a random NDArray of the given shape and dtype with values having a mean and variance.  
  
Parameters:  

- dtype: The data type of the NDArray elements. Defualt: `float64`
  
Args:  

- \*shape: The shape of the NDArray.
- mean: The mean value of the random values. Default: SIMD(0)
- variance: The variance of the random values. Default: SIMD(1)


Example:
    ```py
    var arr = numojo.core.random.rand_meanvar[numojo.i16](3,2,4, mean=0, variance=1)
    print(arr)
    ```


```Mojo
randn[dtype: DType = float64](shape: List[Int], mean: SIMD[dtype, 1] = SIMD(0), variance: SIMD[dtype, 1] = SIMD(1)) -> NDArray[dtype]
```  
Summary  
  
Generate a random NDArray of the given shape and dtype with values having a mean and variance.  
  
Parameters:  

- dtype: The data type of the NDArray elements. Defualt: `float64`
  
Args:  

- shape: The shape of the NDArray.
- mean: The mean value of the random values. Default: SIMD(0)
- variance: The variance of the random values. Default: SIMD(1)


Example:
    ```py
    var arr = numojo.core.random.rand_meanvar[numojo.i16](List[Int](3,2,4), mean=0, variance=1)
    print(arr)
    ```

## rand_exponential


```Mojo
rand_exponential[dtype: DType = float64](*shape: Int, *, rate: SIMD[dtype, 1] = SIMD(#kgen.float_literal<1|1>)) -> NDArray[dtype]
```  
Summary  
  
Generate a random NDArray of the given shape and dtype with values from an exponential distribution.  
  
Parameters:  

- dtype: The data type of the NDArray elements. Defualt: `float64`
  
Args:  

- \*shape: The shape of the NDArray.
- rate: The rate parameter of the exponential distribution (lambda). Default: SIMD(#kgen.float_literal<1|1>)


Example:
    ```py
    var arr = numojo.core.random.rand_exponential[numojo.f64](3, 2, 4, rate=2.0)
    print(arr)
    ```


```Mojo
rand_exponential[dtype: DType = float64](shape: List[Int], rate: SIMD[dtype, 1] = SIMD(#kgen.float_literal<1|1>)) -> NDArray[dtype]
```  
Summary  
  
Generate a random NDArray of the given shape and dtype with values from an exponential distribution.  
  
Parameters:  

- dtype: The data type of the NDArray elements. Defualt: `float64`
  
Args:  

- shape: The shape of the NDArray as a List[Int].
- rate: The rate parameter of the exponential distribution (lambda). Default: SIMD(#kgen.float_literal<1|1>)


Example:
    ```py
    var arr = numojo.core.random.rand_exponential[numojo.f64](List[Int](3, 2, 4), rate=2.0)
    print(arr)
    ```
