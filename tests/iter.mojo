# Test file for `__iter__` method of `NDArray`

import numojo as nm

fn main() raises:
    # test[nm.i8](4)
    test[nm.i8](4, 4)

fn test[dtype: DType](*size: Int) raises:
    var A = nm.NDArray[dtype](size, random=True, order="F")
    print(A)
    print("Iterate over the array:")
    
    for i in A:
        print(i)  # Return 0-d array
    print(str("=") * 30)

    for i in A:
        print(i.item(0))  # Return scalar
    print(str("=") * 30)

    print(A.item(4))