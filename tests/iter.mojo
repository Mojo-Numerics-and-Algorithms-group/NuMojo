# Test file for `__iter__` method of `NDArray`

import numojo as nm

fn main() raises:
    test[nm.i8](10)
    test[nm.f64](20)

fn test[dtype: DType](length: Int) raises:
    var A = nm.NDArray[dtype](length, random=True)
    print(A)
    print("Iterate over the array:")
    for i in A:
        print(i, end="\t")
    print()
    print(str("=") * 30)