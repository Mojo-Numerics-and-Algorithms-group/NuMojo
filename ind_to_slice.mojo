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
    var x = numojo.arange[numojo.f32, numojo.f32](0, 1000, 1)
    var y =  numojo.identity[numojo.i32](10)
    print(x)
    print(y)

    x.reshape(10,10,10)
    print(x[0:2,0:2,1:2])
    print(x[Slice(0,2),Slice(0,2),1])
    print()
    print(x[Slice(0,5),1, 1])
    print(x[0:5, 1:2, 1:2])