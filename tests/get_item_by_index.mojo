import numojo as nm

fn main() raises:
    var A = nm.NDArray[nm.i8](3,random=True)
    print(A)
    print(A[List[Int](2,1,0,1,2)])

    var B = nm.NDArray[nm.i8](3, 3,random=True)
    print(B)
    print(B[List[Int](2,1,0,1,2)])

    var C = nm.NDArray[nm.i8](3, 3, 3,random=True)
    print(C)
    print(C[List[Int](2,1,0,1,2)])


    var X = nm.NDArray[nm.i8](3,random=True)
    print(X)
    print(X.argsort())
    print(X[X.argsort()])
