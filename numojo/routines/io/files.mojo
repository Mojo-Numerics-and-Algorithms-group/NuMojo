from numojo.routines.creation import fromstring
from collections.optional import Optional


fn loadtxt[
    dtype: DType = f64
](
    filename: String,
    delimiter: String = ",",
    skiprows: Int = 0,
    usecols: Optional[List[Int]] = None,
) raises -> NDArray[dtype]:
    with open(filename, "r") as file:
        string = file.read()
        var shape_offset_init: Int = string.find("[")
        var shape_offset_fin: Int = string.find("]")
        var ndim_offset_init: Int = string.find("[", start=shape_offset_fin)
        var ndim_offset_fin: Int = string.find("]", start=ndim_offset_init)
        var ndim: Int = int(string[ndim_offset_init + 1 : ndim_offset_fin])
        var ndshape: List[Int] = List[Int]()
        for i in range(shape_offset_init + 1, shape_offset_fin):
            if string[i].isdigit():
                ndshape.append(int(string[i]))
        var data: List[Scalar[dtype]] = List[Scalar[dtype]]()
        for i in range(ndim_offset_fin + 2, len(string)):
            if string[i].isdigit():
                var number: String = string[i]
                data.append(atof(number).cast[dtype]())
        return array[dtype](data=data, shape=ndshape, order="C")


fn savetxt[
    dtype: DType = f64
](filename: String, array: NDArray[dtype], delimiter: String = ",") raises:
    var shape: String = "ndshape=["
    for i in range(array.ndim):
        shape += str(array.shape[i])
        if i != array.ndim - 1:
            shape = shape + ", "
    shape = shape + "]"
    print(shape)

    with open(filename, "w") as file:
        file.write(shape + "\n")
        file.write("ndim=[" + str(array.ndim) + "]\n")
        for i in range(array.size):
            if i % 10 == 0:
                file.write(str("\n"))
            file.write(str(array._buf[i]) + ",")
