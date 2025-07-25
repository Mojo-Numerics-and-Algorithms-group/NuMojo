from numojo.routines.creation import fromstring
from collections.optional import Optional
from python import Python, PythonObject


# We call into the numpy backend for now, this at least let's people go back and forth smoothly.
# might consider implementing a funciton to write a .numojo file which can be read by both numpy and numojo.


fn load[
    dtype: DType = f64
](
    file: String,
    allow_pickle: Bool = False,
    fix_imports: Bool = True,
    encoding: String = "ASCII",
    *,
    max_header_size: Int = 10000,
) raises -> NDArray[dtype]:
    var np = Python.import_module("numpy")
    var data = np.load(
        file=file,
        allow_pickle=allow_pickle,
        fix_imports=fix_imports,
        encoding=encoding,
        max_header_size=max_header_size,
    )
    var array = numojo.array[dtype](data=data)
    return array^


fn save[
    dtype: DType = f64
](file: String, arr: NDArray[dtype], allow_pickle: Bool = True) raises:
    var np = Python.import_module("numpy")
    var data = np.save(file=file, arr=arr.to_numpy(), allow_pickle=allow_pickle)


fn loadtxt[
    dtype: DType = f64
](
    fname: String,
    comments: String = "#",
    delimiter: String = " ",
    skiprows: Int = 0,
    ndmin: Int = 0,
) raises -> NDArray[dtype]:
    var np = Python.import_module("numpy")
    var data = np.loadtxt(
        fname=fname,
        comments=comments,
        delimiter=delimiter,
        skiprows=skiprows,
        ndmin=ndmin,
    )
    var array = numojo.array[dtype](data=data)
    return array^


fn savetxt[
    dtype: DType = f64
](
    fname: String,
    array: NDArray[dtype],
    fmt: String = "%.18e",
    delimiter: String = " ",
    newline: String = "\n",
    header: String = "",
    footer: String = "",
    comments: String = "#",
) raises:
    var np = Python.import_module("numpy")
    var np_arr = array.to_numpy()
    np.savetxt(
        fname=fname,
        X=np_arr,
        fmt=fmt,
        delimiter=delimiter,
        newline=newline,
        header=header,
        footer=footer,
        comments=comments,
    )
