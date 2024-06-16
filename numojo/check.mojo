import math
import . _math_funcs as _mf
from .ndarray import NDArray

# fn is_power_of_2[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
# ](tensor: NDArray[dtype]) -> NDArray[DType.bool]:
#     return backend()._math_func_is[dtype, math.is_power_of_2](tensor)


# fn is_even[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
# ](tensor: NDArray[dtype]) -> NDArray[DType.bool]:
#     return backend()._math_func_is[dtype, math.is_even](tensor)


# fn is_odd[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
# ](tensor: NDArray[dtype]) -> NDArray[DType.bool]:
#     return backend()._math_func_is[dtype, math.is_odd](tensor)


fn isinf[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: NDArray[dtype]) -> NDArray[DType.bool]:
    return backend()._math_func_is[dtype, math.isinf](tensor)


fn isfinite[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: NDArray[dtype]) -> NDArray[DType.bool]:
    return backend()._math_func_is[dtype, math.isfinite](tensor)


fn isnan[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: NDArray[dtype]) -> NDArray[DType.bool]:
    return backend()._math_func_is[dtype, math.isnan](tensor)
