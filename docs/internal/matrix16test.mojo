import matrix16


fn main() raises:
    var a = matrix16.Matrix16(shape=(6, 6))
    for i in range(a.size):
        (a._buf.get_ptr() + i).init_pointee_copy(i)
    print(a)

    # Get 2-th row as view
    var b = a[2]
    print(b)
    b[0, 0] = -11  # Change the value of view
    print(a)  # The base array also changed

    # Get slice as view
    var c = a[1:6:2, 2:8:2]
    print(c)
    c[1, 1] = -99  # Change the value of view
    print(a)  # The base array also changed

    # Get transpose as view
    # It is a view, so transpose is done in constant time
    var t = matrix16.transpose(a)
    print(t)
    t[2, 3] = -50  # Change the value of view
    print(t)

    # Transpose back
    var tt = matrix16.transpose(t)
    print(tt)

    # Flip by axis
    print(matrix16.flip(a, axis=0))  # Flip by axis 0
    print(matrix16.flip(a, axis=1))  # Flip by axis 1
