# Test file for `__iter__` method of `NDArray`

import numojo as nm

fn main() raises:
    test[nm.i8](3, 3)

fn test[dtype: DType](*size: Int) raises:
    var A = nm.NDArray[dtype](size, random=True, order="F")
    print(A)
    print("Iterate over the array:")
    
    for i in A:
        print(i)  # Return rows
    print(str("=") * 30)

    for i in range(A.size()):
        print(A.item(i))  # Return 0-d arrays
    print(str("=") * 30)