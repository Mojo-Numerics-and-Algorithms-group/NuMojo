from numojo import NDArray

fn sum(array:NDArray, axis:Int)raises->NDArray[array.dtype]:
    var ndim: Int = array.info.ndim
    var shape: List[Int] = array.info.shape
    if axis > ndim-1:
        raise Error("axis cannot be greater than the rank of the array")
    var result_shape: List[Int] = List[Int]()
    var axis_size :Int = shape[axis]
    var slices : List[Slice] = List[Slice]()
    for i in range(ndim):
        if i!=axis:
            result_shape.append(shape[i])
            slices.append(Slice(0,shape[i]))
        else:
            slices.append(Slice(0,0))
    var result: numojo.NDArray[array.dtype] =  NDArray[array.dtype](result_shape)
    
    result+=0
    
    for i in range(axis_size):
        slices[axis] = Slice(i,i+1)
        var arr_slice = array[slices]
        result += arr_slice
    
    return result