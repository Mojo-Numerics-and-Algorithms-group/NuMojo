from numojo.core.ndarray import NDArray
import numojo
# from numojo.math.linalg import matmul_naive, matmul_tiled_unrolled_parallelized, matmul_parallelized
def main():
    # x.reshape(10,10,10)
    # print(x[:2,:2])
    # print(matmul_parallelized(x,x))
    # print(x.info.shape[1])
    # var s = Slice(-1,0)
    # for i in range((s.end-s.start)//s.step):
    #     print(i)
    # var x = numojo.arange[numojo.f32, numojo.f32](0, 1000, 1)
    # var y =  numojo.identity[numojo.i32](10)
    # print(x)
    # print(y)

    # x.reshape(10,10,10)
    # print(x[0:2,0:2,1:2])
    # var temp = x[0:2,0:2,1:2]
    # print(temp)
    # # print(x[Slice(0,2),Slice(0,2),1])
    # print()
    # # print(x[Slice(0,5),1, 1])
    # # print(x[0:5, 1:2, 1:2])
    var arr = NDArray[numojo.f16](VariadicList[Int](2, 3, 3), random=True)
    print("2x3x3 array row major")
    print(arr)
    print()
    # # # # slicing doesn't support single integer index for now, therefore [:,:,2] 
    # # # # should be written as [:,:,2:3] -> both select 2x3 array out of 2x3x3 array
    # # # # * Following are exmaples of slicing operations that can be compared. 

    var sliced = arr[:,:,2:3] # selects 2 x 3 x 1 array out of  2 x 3 x 3
    # var sliced = arr[Slice(0,2),Slice(0,3),2] # selects 2 x 3 x 1 array out of  2 x 3 x 3
    print("2x3 ARRAY - arr[:,:,2:3]")
    print(sliced)
    arr += 10.0
    print(arr)
    print(sliced)

    print(sliced[0,0], sliced[0,1])