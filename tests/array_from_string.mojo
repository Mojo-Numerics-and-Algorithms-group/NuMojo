import numojo as nm

fn main() raises:
    var A = nm.NDArray[DType.int8]("[[[1,2],[3,4]],[[5,6],[7,8]]]")
    var B = nm.NDArray[DType.float16]("[[1,2,3,4],[5,6,7,8]]")
    var C = nm.NDArray("[0.1, -2.3, 41.5, 19.29145, -199]")
    var D = nm.NDArray[DType.int32]("[0.1, -2.3, 41.5, 19.29145, -199]")

    print(A)
    print(B)
    print(C)
    print(D)