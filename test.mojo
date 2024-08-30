# import numojo as nj
import time
from benchmark.compiler import keep
from python import Python
from random import seed

# import numojo as nm
from numojo import *

# alias backend = nm.VectorizedParallelizedNWorkers[8]
# def main():
#     var array = nm.NDArray[nm.f64](10,10, order="F")
#     for i in range(array.size()):
#         array[i]=i
#     var res = array.sum(axis=1)
#     # for i in range(10):
#     #     for j in range(10):
#     #         print(array[i,j])
#     print(res)


fn main() raises:
    var A = arange[i16](1, 7, 1)
    print(A)
    var temp = flip(A)
    print(temp)
    # A.reshape(2, 3, order="F")
    # nm.ravel(A)
    # print(A)
    # var B = arange[i16](0, 12, 1)
    # B.reshape(3, 2, 2, order="F")
    # ravel(B, order="C")
    # print(B)

    # print(A)
    # var B = SIMD[i16, 1](15502)
    # print(B)
    # print(A >= SIMD[i16, 1](10))
    # for i in A:
    #     print(i)

    # var array = arange[i16](0, 12, 1)
    # array.reshape(3, 2, 2)
    # print(array)
    # print("x[0]", array.item(0))
    # print("x[1]", array.item(1))
    # print("x[2]", array.item(2))
    # swapaxis(array, 0, -1)
    # print(array.ndshape)
    # swapaxis(array, 0, 2)
    # print(array.ndshape)
    # moveaxis(array, 0, 2)
    # array.reshape(2, 2, 3, order="C")
    # swapaxis(array, 0, 1)
    # print(array)
    # print("x[0]", array.item(0))
    # print("x[1]", array.item(1))
    # print("x[2]", array.item(2))
    # moveaxis(array, 0, 1)
    # print(array.ndshape)
    # var B = NDArray[i16](3, 3, random=True)
    # print(B)
    # A[A > 10.0] = B
    # print(A)
    # var gt = A > 10.0
    # print(gt)
    # var ge = A >= Scalar[i16](10)
    # print(ge)
    # var lt = A < Scalar[i16](10)
    # print(lt)
    # var le = A <= Scalar[i16](10)
    # print(le)
    # var eq = A == Scalar[i16](10)
    # print(eq)
    # var ne = A != Scalar[i16](10)
    # print(ne)
    # var mask = A[A > Scalar[i16](10)]
    # print(mask)
    # seed(10)
    # var B = NDArray[i16](3, 3, random=True)
    # print(B)
    # var gta = A > B
    # print(gta)
    # var gte = A >= B
    # print(gte)
    # var lt = A < B
    # print(lt)
    # var le = A <= B
    # print(le)
    # var eq = A == B
    # print(eq)
    # var ne = A != B
    # print(ne)
    # var mask = A[A > B]
    # print(mask)
    # var temp = A * SIMD[i8, 1](2)
    # print(temp)
    # var temp1 = temp == 0
    # var temp2 = A[temp]
    # print(temp2)
    # var temp2 = A[A%2 == 0]
    # print(temp2)
    # print(temp2.at(0))
    # print(temp2.ndshape, temp2.stride, temp2.ndshape._size)


# def main():

# # CONSTRUCTORS TEST
# var arr1 = numojo.NDArray[numojo.f32](3,4,5, random=True)
# print(arr1)
# print("ndim: ", arr1.ndim)
# print("shape: ", arr1.ndshape)
# print("strides: ", arr1.stride)
# print("size: ", arr1.ndshape.ndsize)
# print("offset: ", arr1.stride.ndoffset)
# print("dtype: ", arr1.dtype)

# print()

# var arr2 = numojo.NDArray[numojo.f32](VariadicList[Int](3, 4, 5), random=True)
# print(arr2)
# print("ndim: ", arr2.ndim)
# print("shape: ", arr2.ndshape)
# print("strides: ", arr2.stride)
# print("size: ", arr2.ndshape.ndsize)
# print("offset: ", arr2.stride.ndoffset)
# print("dtype: ", arr2.dtype)

# var arr3 = numojo.NDArray[numojo.f32](VariadicList[Int](3, 4, 5), fill = 10.0)
# print(arr3)
# print("ndim: ", arr3.ndim)
# print("shape: ", arr3.ndshape)
# print("strides: ", arr3.stride)
# print("size: ", arr3.ndshape.ndsize)
# print("offset: ", arr3.stride.ndoffset)
# print("dtype: ", arr3.dtype)

# var arr4 = numojo.NDArray[numojo.f32](List[Int](3, 4, 5), random=True)
# print(arr4)
# print("ndim: ", arr4.ndim)
# print("shape: ", arr4.ndshape)
# print("strides: ", arr4.stride)
# print("size: ", arr4.ndshape.ndsize)
# print("offset: ", arr4.stride.ndoffset)
# print("dtype: ", arr4.dtype)

# var arr5 = numojo.NDArray[numojo.f32](numojo.NDArrayShape(3,4,5), random=True)
# print(arr5)
# print("ndim: ", arr5.ndim)
# print("shape: ", arr5.ndshape)
# print("strides: ", arr5.stride)
# print("size: ", arr5.ndshape.ndsize)
# print("offset: ", arr5.stride.ndoffset)
# print("dtype: ", arr5.dtype)

# var arr6 = numojo.NDArray[numojo.f32](data=List[SIMD[numojo.f32, 1]](1,2,3,4,5,6,7,8,9,10), shape=
# List[Int](2,5))
# print(arr6)
# print("ndim: ", arr6.ndim)
# print("shape: ", arr6.ndshape)
# print("strides: ", arr6.stride)
# print("size: ", arr6.ndshape.ndsize)
# print("offset: ", arr6.stride.ndoffset)
# print("dtype: ", arr6.dtype)

# var x = numojo.linspace[numojo.f32](0.0, 60.0, 60)
# var x = numojo.ones[numojo.f32](3, 2)
# var x = numojo.logspace[numojo.f32](-3, 0, 60)
# var x = arange[f32](0.0, 24.0, step=1)
# x.reshape(2, 3, 4, order="C")
# print(x[0,0,0], x[0,0,1], x[1,1,1], x[1,2,3])
# print(x)
# var slicedx = x[0:1, :, 1:2]
# print(slicedx)
# print()

# var y = numojo.arange[numojo.f32](0.0, 24.0, step=1)
# y.reshape(2,3,4, order="F")
# print(y[0,0,0], y[0,0,1], y[1,1,1], y[1,2,3])
# print(y)
# print(y.order)
# var slicedy = y[:, :, 1:2]
# print(slicedy)
# var x = numojo.full[numojo.f32](3, 2, fill_value=16.0)
# var x = numojo.NDArray[numojo.f32](data=List[SIMD[numojo.f32, 1]](1,2,3,4,5,6,7,8,9,10,11,12), shape=List[Int](2,3,2),
# order="F")
# print(x)
# print(x.stride)
# var y = numojo.NDArray[numojo.f32](data=List[SIMD[numojo.f32, 1]](1,2,3,4,5,6,7,8,9,10,11,12), shape=List[Int](2,3,2),
# order="C")
# print(y)
# print(y.stride)
# print()
# var summed = numojo.stats.sum(x,0)
# print(summed)
# print(numojo.stats.mean(x,0))
# print(numojo.stats.cumprod(x))

# var maxval = x.max(axis=0)
# print(maxval)

# var arr = numojo.NDArray[numojo.f32](data=List[SIMD[numojo.f32, 1]](1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0),
# shape=List[Int](3,3), order="C")
# var raw = List[Int32]()
# for _ in range(16):
#     raw.append(random.randn_float64()*10)
# var arr1 = numojo.NDArray[numojo.i32](data=raw, shape=List[Int](4,4), order="C")
# var arr = numojo.NDArray[DType.int8](3,3,3, random=True, order="C")
# var arr1 = numojo.NDArray[DType.bool](data=List[SIMD[DType.bool, 1]](False, False, False, True),
# shape=List[Int](2,2))
# print(arr1)
# print(arr1[0,1])
# print(arr1[0:1, :])
# print(arr1[1:2, 3:4])

# var array = nj.NDArray[nj.f64](10,10)
# for i in range(array.size()):
#     array[i] = i
# # for i in range(10):
#     # for j in range(10):
#         # print(array[i, j])
# var res = array.sum(axis=0)
# print(res)

# var arr2 = numojo.NDArray[numojo.f32](data=List[SIMD[numojo.f32, 1]](1.0, 2.0, 4.0, 7.0, 11.0, 16.0),
# shape=List[Int](6))
# var np = Python.import_module("numpy")
# var np_arr = numojo.to_numpy(arr2)
# print(np_arr)
# var result = numojo.math.calculus.differentiation.gradient[numojo.f32](arr2, spacing=1.0)
# print(result)
# print(arr1.any())
# print(arr1.all())
# print(arr1.argmax())
# print(arr1.argmin())
# print(arr1.astype[numojo.i16]())
# print(arr1.flatten(inplace=True))
# print(r.ndshape, r.stride, r.ndshape.ndsize)
# var t0 = time.now()
# var res = numojo.math.linalg.matmul_tiled_unrolled_parallelized[numojo.f32](arr, arr1)
# print((time.now()-t0)/1e9)
# var res = numojo.math.linalg.matmul_tiled_unrolled_parallelized[numojo.f32](arr, arr1)
# print(res)
# print(arr)
# print("2x3x1")
# var sliced = arr[:, :, 1:2]
# print(sliced)

# print("1x3x4")
# var sliced1 = arr[::2, :]
# print(sliced1)

# print("1x3x1")
# var sliced2 = arr[1:2, :, 2:3]
# print(sliced2)

# var result = numojo.NDArray(3, 3)
# numojo.math.linalg.dot[t10=3, t11=3, t21=3, dtype=numojo.f32](result, arr, arr1)
# print(result)

# fn main() raises:
#     var size:VariadicList[Int] = VariadicList[Int](16,128,256,512,1024)
#     alias size1: StaticIntTuple[5] = StaticIntTuple[5](16,128,256,512,1024)
#     var times:List[Float64] = List[Float64]()
#     alias type:DType = DType.float64
#     measure_time[type, size1](size, times)

# fn measure_time[dtype:DType, size1: StaticIntTuple[5]](size:VariadicList[Int], inout times:List[Float64]) raises:

#     for i in range(size.__len__()):
#         var arr1 = numojo.NDArray[dtype](size[i], size[i], random=True)
#         var arr2 = numojo.NDArray[dtype](size[i], size[i], random=True)
#         var arr_mul = numojo.NDArray[dtype](size[i], size[i])

#         var t0 = time.now()
#         @parameter
#         for i in range(50):
#             numojo.math.linalg.dot[t10=size1[i], t11=size1[i], t21=size1[i], dtype=dtype](arr_mul, arr1, arr2)
#             # var arr_mul = numojo.math.linalg.matmul_parallelized[dtype](arr1, arr2)
#             # var arr_mul = numojo.math.linalg.matmul_tiled_unrolled_parallelized[dtype](arr1, arr2)
#             keep(arr_mul.unsafe_ptr())
#         times.append(((time.now()-t0)/1e9)/50)

#     for i in range(size.__len__()):
#         print(times[i])

# fn main() raises:
#     alias type:DType = DType.float16
#     measure_time[type]()

# fn measure_time[dtype:DType]() raises:
#     var size:VariadicList[Int] = VariadicList[Int](16,128,256,512,1024)
#     alias size1: StaticIntTuple[5] = StaticIntTuple[5](16,128,256,512,1024)

#     var n = 4
#     alias m = 4
#     var arr1 = numojo.NDArray[dtype](size[n], size[n], random=True)
#     var arr2 = numojo.NDArray[dtype](size[n], size[n], random=True)
#     var arr_mul = numojo.NDArray[dtype](size[n], size[n])

#     var t0 = time.now()

#     for _ in range(50):
#         numojo.math.linalg.dot[t10=size1[m], t11=size1[m], t21=size1[m], dtype=dtype](arr_mul, arr1, arr2)
#         # var arr_mul = numojo.math.linalg.matmul_parallelized[dtype](arr1, arr2)
#         # var arr_mul = numojo.math.linalg.matmul_tiled_unrolled_parallelized[dtype](arr1, arr2)
#         keep(arr_mul.unsafe_ptr())
#     print(((time.now()-t0)/1e9)/50)
