# Test file for
# numojo.math.statistics.stats.mojo

import numojo as nm
from numojo.core.ndarray import NDArray
import time

fn main() raises:
    var A = NDArray(3, 3, random=True)
    print(A)
    print(nm.math.stats.sumall(A))

    print(A)
    print(nm.math.stats.prodall(A))

    print(A)
    print(nm.math.stats.meanall(A))

    print(A)
    print(nm.math.stats.max(A))