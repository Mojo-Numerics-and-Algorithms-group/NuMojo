from algorithm import parallelize

from tensor import Tensor, TensorShape

from builtin.math import pow

################### ARANGE ####################
fn arange[dtype:DType](
        start: Scalar[dtype], 
        stop: Scalar[dtype], 
        step: Scalar[dtype]) -> Tensor[dtype]:
    """
    Function that computes a series of values starting from "start" to "stop" with given "step" size.

    Parameter:
        dtype: DType  - datatype of the Tensor

    Args:
        start: Scalar[dtype] - Start value.
        stop: Scalar[dtype]  - End value.
        step: Scalar[dtype]  - Step size between each element.

    Returns:
        Tensor[dtype] - Tensor of datatype T with elements ranging from "start" to "stop" incremented with "step".
    """
    var num: Int = ((stop - start) / step).__int__()
    var result: Tensor[dtype] = Tensor[dtype](TensorShape(num))
    for idx in range(num):
        result[idx] = start + step * idx
    return result

# I think defaulting parallelization to False is better
#################### LINEAR SPACE ############
fn linspace[dtype:DType](
        start: Scalar[dtype], 
        stop: Scalar[dtype], 
        num: Int, 
        endpoint: Bool = True, 
        parallel: Bool = False) -> Tensor[dtype]:
    """
    Function that computes a series of linearly spaced values starting from "start" to "stop" with given size. Wrapper function for _linspace_serial, _linspace_parallel.

    Parameter:
        dtype: DType - datatype of the Tensor

    Args:
        start: Scalar[dtype] - Start value.
        stop: Scalar[dtype]  - End value.
        num: Int  - No of linearly spaced elements.
        endpoint: Bool - Specifies whether to include endpoint in the final tensor, defaults to True.
        parallel: Bool - Specifies whether the linspace should be calculated using parallelization, deafults to False.

    Returns:
        Tensor[dtype] - Tensor of datatype T with elements ranging from "start" to "stop" with num elements. 

    """
    if parallel:
        return _linspace_parallel[dtype](start, stop, num, endpoint)
    else:
        return _linspace_serial[dtype](start, stop, num, endpoint)

fn _linspace_serial[dtype:DType](
        start: Scalar[dtype], 
        stop: Scalar[dtype], 
        num: Int, 
        endpoint: Bool = True) -> Tensor[dtype]:
    """
    Generate a linearly spaced tensor of `num` elements between `start` and `stop` using naive for loop.

    Parameters:
        dtype: DType - datatype of the Tensor
    
    Args:
        start: The starting value of the tensor.
        stop: The ending value of the tensor.
        num: The number of elements in the tensor.
        endpoint: Whether to include the `stop` value in the tensor. Defaults to True.

    Returns:
    - A tensor of `dtype` with `num` linearly spaced elements between `start` and `stop`.
    """
    var result: Tensor[dtype] = Tensor[dtype](TensorShape(num))

    if endpoint:
        var step: Scalar[dtype] = (stop - start) / (num - 1)
        for i in range(num):
            result[i] = start + step * i

    else:
        var step: Scalar[dtype] = (stop - start) / num
        for i in range(num):
            result[i] = start + step * i

    return result

fn _linspace_parallel[dtype:DType](
        start: Scalar[dtype], 
        stop: Scalar[dtype], 
        num: Int, 
        endpoint: Bool = True) -> Tensor[dtype]:
    """
    Generate a linearly spaced tensor of `num` elements between `start` and `stop` using parallelization.

    Parameters:
        dtype: DType - datatype of the Tensor
    
    Args:
        start: The starting value of the tensor.
        stop: The ending value of the tensor.
        num: The number of elements in the tensor.
        endpoint: Whether to include the `stop` value in the tensor. Defaults to True.

    Returns:
    - A tensor of `dtype` with `num` linearly spaced elements between `start` and `stop`.
    """
    var result: Tensor[dtype] = Tensor[dtype](TensorShape(num))
    alias nelts = simdwidthof[dtype]()

    if endpoint:
        var step: Scalar[dtype] = (stop - start) / (num - 1)
        @parameter
        fn parallelized_linspace(idx: Int) -> None:
            # result.store[width=nelts](idx, start + step*idx)
            result[idx] = start + step * idx
        parallelize[parallelized_linspace](num)

    else:
        var step: Scalar[dtype] = (stop - start) / num
        @parameter
        fn parallelized_linspace1(idx: Int) -> None:
            # result.store[width=nelts](idx, start + step*idx)
            result[idx] = start + step * idx
        parallelize[parallelized_linspace1](num)

    return result

#################### LOGSPACE ################
fn logspace[dtype:DType](
        start: Scalar[dtype], 
        stop: Scalar[dtype], 
        num: Int, 
        endpoint: Bool = True, 
        base:Scalar[dtype]=10.0, 
        parallel: Bool = False) -> Tensor[dtype]:
    """
    Generate a logrithmic spaced tensor of `num` elements between `start` and `stop`. Wrapped function for _logspace_serial, _logspace_parallel funtions.

    Parameters:
        dtype: DType - datatype of the Tensor.
    
    Args:
        start: The starting value of the tensor.
        stop: The ending value of the tensor.
        num: The number of elements in the tensor.
        endpoint: Whether to include the `stop` value in the tensor. Defaults to True.
        base: Base value of the logarithm, defaults to 10.
        parallel: Specifies whether to calculate the logarithmic spaced values using parallelization.

    Returns:
    - A tensor of `dtype` with `num` logaritmic spaced elements between `start` and `stop`.
    """
    if parallel:
        return _logspace_parallel[dtype](start, stop, num, base, endpoint)
    else:
        return _logspace_serial[dtype](start, stop, num, base, endpoint)

fn _logspace_serial[dtype:DType](
        start: Scalar[dtype], 
        stop: Scalar[dtype], 
        num: Int, 
        base:Scalar[dtype], 
        endpoint: Bool = True) -> Tensor[dtype]:
    """
    Generate a logarithmic spaced tensor of `num` elements between `start` and `stop` using naive for loop.

    Parameters:
        dtype: DType - datatype of the Tensor
    
    Args:
        start: The starting value of the tensor.
        stop: The ending value of the tensor.
        num: The number of elements in the tensor.
        base: Base value of the logarithm, defaults to 10.
        endpoint: Whether to include the `stop` value in the tensor. Defaults to True.

    Returns:
    - A tensor of `dtype` with `num` logarithmic spaced elements between `start` and `stop`.
    """ 
    var result: Tensor[dtype] = Tensor[dtype](TensorShape(num))

    if endpoint:
        var step: Scalar[dtype] = (stop - start) / (num - 1)
        for i in range(num):
            result[i] = base**(start + step * i)
    else:
        var step: Scalar[dtype] = (stop - start) / num
        for i in range(num):
            result[i] = base**(start + step * i)
    return result

fn _logspace_parallel[dtype:DType](
        start: Scalar[dtype], 
        stop: Scalar[dtype], 
        num: Int, 
        base:Scalar[dtype], 
        endpoint: Bool = True) -> Tensor[dtype]:
    """
    Generate a logarithmic spaced tensor of `num` elements between `start` and `stop` using parallelization.

    Parameters:
        dtype: DType - datatype of the Tensor
    
    Args:
        start: The starting value of the tensor.
        stop: The ending value of the tensor.
        num: The number of elements in the tensor.
        base: Base value of the logarithm, defaults to 10.
        endpoint: Whether to include the `stop` value in the tensor. Defaults to True.

    Returns:
    - A tensor of `dtype` with `num` linearly spaced elements between `start` and `stop`.
    """
    var result: Tensor[dtype] = Tensor[dtype](TensorShape(num))

    if endpoint:
        var step: Scalar[dtype] = (stop - start) / (num - 1)
        @parameter
        fn parallelized_linspace(idx: Int) -> None:
            result[idx] = base**(start + step * idx)
        parallelize[parallelized_linspace](num)

    else:
        var step: Scalar[dtype] = (stop - start) / num
        @parameter
        fn parallelized_linspace1(idx: Int) -> None:
            result[idx] = base**(start + step * idx)
        parallelize[parallelized_linspace1](num)

    return result

# ! Outputs wrong values for Integer type, works fine for float type. 
fn geomspace[dtype:DType](
        start: Scalar[dtype], 
        stop: Scalar[dtype], 
        num: Int, 
        endpoint: Bool = True) -> Tensor[dtype]:
    """
    Generate a tensor of `num` elements between `start` and `stop` in a geometric series.

    Parameters:
        dtype: DType - datatype of the Tensor.
    
    Args:
        start: The starting value of the tensor.
        stop: The ending value of the tensor.
        num: The number of elements in the tensor.
        endpoint: Whether to include the `stop` value in the tensor. Defaults to True.

    Returns:
    - A tensor of `dtype` with `num` geometrically spaced elements between `start` and `stop`. 
    """
    var a:Scalar[dtype] = start

    if endpoint:
        var result:Tensor[dtype] = Tensor[dtype](TensorShape(num))
        # var r:Scalar[dtype] = math.pow(stop/start, 1/(num-1))
        var r: Scalar[dtype] = (stop/start)**(1/(num-1))
        for i in range(num):
            result[i] = a * r**i
        return result

    else:
        var result:Tensor[dtype] = Tensor[dtype](TensorShape(num))
        var r:Scalar[dtype] = (stop/start)**(1/(num-1)) 
        for i in range(num):
            result[i] = a * r**i 
        return result
    
fn zeros[dtype:DType](
        *shape:Int) -> Tensor[dtype]:
    """
    Generate a tensor of zeros with given shape.

    Parameters:
        dtype: DType - datatype of the Tensor.
    
    Args:
        shape: VariadicList[Int] - Shape of the tensor.

    Returns:
    - A tensor of `dtype` with given `shape`.
    """
    var tens_shape:VariadicList[Int] = shape
    return Tensor[dtype](tens_shape)

fn eye[dtype:DType](
        N:Int, 
        M:Int, 
        k:Int=0) -> Tensor[dtype]:
    """
    Generate an identity matrix of size N x M.

    Parameters:
        dtype: DType - datatype of the Tensor.
    
    Args:
        N: Int - Number of rows in the matrix.
        M: Int - Number of columns in the matrix.
        k: Int - Number of diagonals to be filled with ones. Defaults to 0.

    Returns:
    - A tensor of `dtype` with size N x M and ones on the diagonals.
    """
    var result:Tensor[dtype] = Tensor[dtype](N,M)
    var one = Scalar[dtype](1)
    # for i in range(N):
    #     for j in range(M):
    #         if i == j:
    #             # result[i*M + j] = one
    #             result[VariadicList[Int](i,j)] = one
    #         else:
    #             continue
    for idx in range(M*N):
        if (idx + 1 - M) // N == 0:
            result[idx] = one
        else:
            continue
    return result

fn identity[dtype:DType](
        n:Int) -> Tensor[dtype]:
    """
    Generate an identity matrix of size N x N.

    Parameters:
        dtype: DType - datatype of the Tensor.
    
    Args:
        n: Int - Size of the matrix.

    Returns:
    - A tensor of `dtype` with size N x N and ones on the diagonals.
    """
    return eye[dtype](n,n)

fn ones[dtype:DType](
        *shape:Int) -> Tensor[dtype]:
    """
    Generate a tensor of ones with given shape.

    Parameters:
        dtype: DType - datatype of the Tensor.
    
    Args:
        shape: VariadicList[Int] - Shape of the tensor.

    Returns:
    - A tensor of `dtype` with given `shape`.
    """
    var tens_shape:VariadicList[Int] = shape
    return Tensor[dtype](tens_shape, Scalar[dtype](1))

fn fill[dtype:DType,
        fill_value:Scalar[dtype]
        ](*shape:Int) -> Tensor[dtype]:
    """
    Generate a tensor of `fill_value` with given shape.

    Parameters:
        dtype: DType - datatype of the Tensor.
        fill_value: Scalar[dtype] - value to be splatted over the tensor.
    
    Args:
        shape: VariadicList[Int] - Shape of the tensor.

    Returns:
    - A tensor of `dtype` with given `shape`.
    """
    var tens_shape:VariadicList[Int] = shape
    return Tensor[dtype](tens_shape, fill_value)

fn fill[dtype:DType](
        shape:VariadicList[Int], 
        fill_value:Scalar[dtype]) -> Tensor[dtype]:
    """
    Generate a tensor of `fill_value` with given shape.

    Parameters:
        dtype: DType - datatype of the Tensor.
    
    Args:
        shape: VariadicList[Int] - Shape of the tensor.
        fill_value: Scalar[dtype] - value to be splatted over the tensor.

    Returns:
    - A tensor of `dtype` with given `shape`.
    """
    return Tensor[dtype](shape, fill_value)