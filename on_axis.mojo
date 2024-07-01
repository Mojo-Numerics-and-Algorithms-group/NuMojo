import numojo
from numojo import *
from time import now
from numojo.math.statistics import mean, stdev,sum
from benchmark.compiler import keep
fn on_axis(array:numojo.NDArray, axis:Int)raises->numojo.NDArray[array.dtype]:
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
    var result: numojo.NDArray[array.dtype] =  numojo.NDArray[array.dtype](result_shape)
    
    result+=0 # This prevents a segfualt for some reason
    
    for i in range(axis_size):
        slices[axis] = Slice(i,i+1)
        var arr_slice = array[slices]
        # print(arr_slice.shape())
        result += arr_slice
    
    return result
            
fn sum_on_axis(array:NDArray,axis:Int)raises->NDArray[array.dtype]:
    var stride_size: Int = array.info.strides[axis]
    if axis!=0:
        var stride_size_off: Int = array.info.strides[axis]
    var ndim: Int = array.info.ndim
    var shape: List[Int] = array.info.shape
    if axis > ndim-1:
        raise Error("axis cannot be greater than the rank of the array")
    var result_shape: List[Int] = List[Int]()
    
    for i in range(ndim):
        if i!=axis:
            result_shape.append(shape[i])
    var size_at_axis = shape[axis]
    var result: numojo.NDArray[array.dtype] =  numojo.NDArray[array.dtype](result_shape)
    alias opt_nelts = simdwidthof[array.dtype]()
    result+=0
    for i in range(result.info.size):
        var simd = SIMD[array.dtype,1](0)
        var offset=0
        if axis!=0:
            var stride_size_off: Int = array.info.strides[axis-1]
            offset = stride_size_off*(i//result.info.shape[0])
        print(offset)
        for j in range(size_at_axis//opt_nelts):
            simd+=(array._arr+i+j*(stride_size)).simd_strided_load[opt_nelts](stride_size).reduce_add()
        for j in range(opt_nelts*(size_at_axis//opt_nelts),size_at_axis):
            simd += (array._arr+i+j*stride_size).simd_strided_load[1](stride_size) 
        result.store[1](i,simd)
    return result



def main():
    alias f32array = numojo.NDArray[numojo.f32]
    var ones = numojo.ones[numojo.f32](4,5,3)
    var x = numojo.arange[numojo.f32](0,60)
    # var x = numojo.zeros[f32](4,5,3)
    # for i in range(60):
        # x[i] = i
    x.reshape(4,5,3)
    # print(ones)
    print(numojo.stats.mean(x,0))
    # print(sum_on_axis(x,0))
    # print()
    # print(on_axis(x,1))
    # x+=0
    # print(on_axis(ones,0))
    # print(x._arr.simd_strided_load[8](101))
    # print((x._arr+808).simd_strided_load[2](101))
    # print((x._arr+909).simd_strided_load[1](101))
    for i in range(len(x.info.coefficients)):
        print(x.info.strides[i])
    # res = numojo.zeros[numojo.f32](10,10)
    # o1 = ones[Slice(0,0+1),Slice(0,10),Slice(0,10)]
    # o2 = ones[Slice(0,0+1),Slice(0,10),Slice(0,10)]
    # # print(o1+o2)
    # for i in range(ones.info.shape[0]):
    # #     print(ones[Slice(0,0+1),Slice(0,10),Slice(0,10)])
    #     var o1 =ones[Slice(i,i+1),Slice(0,10),Slice(0,10)]
    #     res = res + o1
    # print(res)
    # ones+=0
    # var times = empty[f32](250)
    # for i in range(250):
    #     var t1 = now()
    #     try:
    #         var res = sum_on_axis(x,0)
    #         keep(res.unsafe_ptr())
    #     except:
    #         print('fma: Failed shape error')
    #     var t2 = now()
    #     times[i] = t2-t1
    # print("mean: ", mean[f32](times[10:]))
    # print("std: ", stdev[f32](times[10:]))