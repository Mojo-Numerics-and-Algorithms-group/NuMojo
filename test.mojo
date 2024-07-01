import numojo

def main():

    # # CONSTRUCTORS TEST
    # var arr1 = numojo.NDArray[numojo.f32](3,4,5, random=True)
    # print(arr1)
    # print("ndim: ", arr1.ndim)
    # print("shape: ", arr1.ndshape)
    # print("strides: ", arr1.stride)
    # print("size: ", arr1.ndshape._size)
    # print("offset: ", arr1.stride._offset)
    # print("dtype: ", arr1.dtype)

    # print()

    # var arr2 = numojo.NDArray[numojo.f32](VariadicList[Int](3, 4, 5), random=True)
    # print(arr2)
    # print("ndim: ", arr2.ndim)
    # print("shape: ", arr2.ndshape)
    # print("strides: ", arr2.stride)
    # print("size: ", arr2.ndshape._size)
    # print("offset: ", arr2.stride._offset)
    # print("dtype: ", arr2.dtype)
 
    # var arr3 = numojo.NDArray[numojo.f32](VariadicList[Int](3, 4, 5), val = 10.0)
    # print(arr3)
    # print("ndim: ", arr3.ndim)
    # print("shape: ", arr3.ndshape)
    # print("strides: ", arr3.stride)
    # print("size: ", arr3.ndshape._size)
    # print("offset: ", arr3.stride._offset)
    # print("dtype: ", arr3.dtype)

    # var arr4 = numojo.NDArray[numojo.f32](List[Int](3, 4, 5), random=True)
    # print(arr4)
    # print("ndim: ", arr4.ndim)
    # print("shape: ", arr4.ndshape)
    # print("strides: ", arr4.stride)
    # print("size: ", arr4.ndshape._size)
    # print("offset: ", arr4.stride._offset)
    # print("dtype: ", arr4.dtype) 

    # var arr5 = numojo.NDArray[numojo.f32](numojo.NDArrayShape(3,4,5), random=True)
    # print(arr5)
    # print("ndim: ", arr5.ndim)
    # print("shape: ", arr5.ndshape)
    # print("strides: ", arr5.stride)
    # print("size: ", arr5.ndshape._size)
    # print("offset: ", arr5.stride._offset)
    # print("dtype: ", arr5.dtype) 

    # var arr6 = numojo.NDArray[numojo.f32](data=List[SIMD[numojo.f32, 1]](1,2,3,4,5,6,7,8,9,10), shape=
    # List[Int](2,5))
    # print(arr6)
    # print("ndim: ", arr6.ndim)
    # print("shape: ", arr6.ndshape)
    # print("strides: ", arr6.stride)
    # print("size: ", arr6.ndshape._size)
    # print("offset: ", arr6.stride._offset)
    # print("dtype: ", arr6.dtype) 


    # var x = numojo.linspace[numojo.f32](0.0, 60.0, 60)
    # var x = numojo.ones[numojo.f32](3, 2)
    # var x = numojo.logspace[numojo.f32](-3, 0, 60)
    var x = numojo.arange[numojo.f32](0.0, 60.0, step=1)
    print(x)
    x.reshape(4,5,3)
    print(x)
    print()
    var summed = numojo.stats.sum(x,0)
    print(summed)
    print(numojo.stats.mean(x,0))

    # var arr = numojo.NDArray[numojo.f32](2,3,3, random=True)
    # print(arr)
    # var sliced = arr[0:1,:,:]
    # print(sliced)
    # var sliced1 = arr[:,1:2,:]
    # print(sliced1)