from numojo.prelude import *
from numojo.core.memory.dlpack import from_dlpack
from python import Python
import benchmark


def main() raises:
    var np = Python.import_module("numpy")
    var numpy_data = np.linspace(0, 5, 6, dtype=np.float32)

    def make_view() raises capturing -> None:
        var mojo_arr = from_dlpack[f32](numpy_data)

    var report = benchmark.run[make_view]()
    report.print()
