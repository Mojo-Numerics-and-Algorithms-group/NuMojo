from builtin.math import pow
import math as mt
from utils.numerics import isnan, isinf

from numojo.core.utility import is_inttype, is_floattype

alias DEFAULT_PRECISION = 4
alias DEFAULT_SUPPRESS_SMALL = False
alias DEFAULT_SEPARATOR = " "
alias DEFAULT_PADDING = ""
alias DEFAULT_EDGE_ITEMS = 3
alias DEFAULT_THRESHOLD = 10
alias DEFAULT_LINE_WIDTH = 75
alias DEFAULT_SIGN = False
alias DEFAULT_FLOAT_FORMAT = "fixed"
alias DEFAULT_COMPLEX_FORMAT = "parentheses"
alias DEFAULT_NAN_STRING = "nan"
alias DEFAULT_INF_STRING = "inf"
alias DEFAULT_FORMATTED_WIDTH = 8
alias DEFAULT_EXPONENT_THRESHOLD = 4
alias DEFAULT_SUPPRESS_SCIENTIFIC = False

var GLOBAL_PRINT_OPTIONS: PrintOptions = PrintOptions(
    precision=DEFAULT_PRECISION,
    suppress_small=DEFAULT_SUPPRESS_SMALL,
    separator=DEFAULT_SEPARATOR,
    padding=DEFAULT_PADDING,
    threshold=DEFAULT_THRESHOLD,
    line_width=DEFAULT_LINE_WIDTH,
    edge_items=DEFAULT_EDGE_ITEMS,
    sign=DEFAULT_SIGN,
    float_format=DEFAULT_FLOAT_FORMAT,
    complex_format=DEFAULT_COMPLEX_FORMAT,
    nan_string=DEFAULT_NAN_STRING,
    inf_string=DEFAULT_INF_STRING,
    formatted_width=DEFAULT_FORMATTED_WIDTH,
    exponent_threshold=DEFAULT_EXPONENT_THRESHOLD,
    suppress_scientific=DEFAULT_SUPPRESS_SCIENTIFIC,
)

alias printoptions = PrintOptions


@value
struct PrintOptions:
    var precision: Int
    var suppress_small: Bool
    var separator: String
    var padding: String
    var threshold: Int
    var line_width: Int
    var edge_items: Int
    var sign: Bool
    var float_format: String
    var complex_format: String
    var nan_string: String
    var inf_string: String
    var formatted_width: Int
    var exponent_threshold: Int
    var suppress_scientific: Bool

    fn __init__(
        mut self,
        precision: Int = DEFAULT_PRECISION,
        suppress_small: Bool = DEFAULT_SUPPRESS_SMALL,
        separator: String = DEFAULT_SEPARATOR,
        padding: String = DEFAULT_PADDING,
        threshold: Int = DEFAULT_THRESHOLD,
        line_width: Int = DEFAULT_LINE_WIDTH,
        edge_items: Int = DEFAULT_EDGE_ITEMS,
        sign: Bool = DEFAULT_SIGN,
        float_format: String = DEFAULT_FLOAT_FORMAT,
        complex_format: String = DEFAULT_COMPLEX_FORMAT,
        nan_string: String = DEFAULT_NAN_STRING,
        inf_string: String = DEFAULT_INF_STRING,
        formatted_width: Int = DEFAULT_FORMATTED_WIDTH,
        exponent_threshold: Int = DEFAULT_EXPONENT_THRESHOLD,
        suppress_scientific: Bool = DEFAULT_SUPPRESS_SCIENTIFIC,
    ):
        self.precision = precision
        self.suppress_small = suppress_small
        self.separator = separator
        self.padding = padding
        self.threshold = threshold
        self.line_width = line_width
        self.edge_items = edge_items
        self.sign = sign
        self.float_format = float_format
        self.complex_format = complex_format
        self.nan_string = nan_string
        self.inf_string = inf_string
        self.formatted_width = formatted_width
        self.exponent_threshold = exponent_threshold
        self.suppress_scientific = suppress_scientific

    fn set_options(
        mut self,
        precision: Int = DEFAULT_PRECISION,
        suppress_small: Bool = DEFAULT_SUPPRESS_SMALL,
        separator: String = DEFAULT_SEPARATOR,
        padding: String = DEFAULT_PADDING,
        threshold: Int = DEFAULT_THRESHOLD,
        line_width: Int = DEFAULT_LINE_WIDTH,
        edge_items: Int = DEFAULT_EDGE_ITEMS,
        sign: Bool = DEFAULT_SIGN,
        float_format: String = DEFAULT_FLOAT_FORMAT,
        complex_format: String = DEFAULT_COMPLEX_FORMAT,
        nan_string: String = DEFAULT_NAN_STRING,
        inf_string: String = DEFAULT_INF_STRING,
        formatted_width: Int = DEFAULT_FORMATTED_WIDTH,
        exponent_threshold: Int = DEFAULT_EXPONENT_THRESHOLD,
        suppress_scientific: Bool = DEFAULT_SUPPRESS_SCIENTIFIC,
    ):
        self.precision = precision
        self.suppress_small = suppress_small
        self.separator = separator
        self.padding = padding
        self.threshold = threshold
        self.line_width = line_width
        self.edge_items = edge_items
        self.sign = sign
        self.float_format = float_format
        self.complex_format = complex_format
        self.nan_string = nan_string
        self.inf_string = inf_string
        self.formatted_width = formatted_width
        self.exponent_threshold = exponent_threshold
        self.suppress_scientific = suppress_scientific

    fn __enter__(mut self) -> Self:
        GLOBAL_PRINT_OPTIONS.set_options(
            precision=self.precision,
            suppress_small=self.suppress_small,
            separator=self.separator,
            padding=self.padding,
            threshold=self.threshold,
            line_width=self.line_width,
            edge_items=self.edge_items,
            sign=self.sign,
            float_format=self.float_format,
            complex_format=self.complex_format,
            nan_string=self.nan_string,
            inf_string=self.inf_string,
            formatted_width=self.formatted_width,
            exponent_threshold=self.exponent_threshold,
            suppress_scientific=self.suppress_scientific,
        )
        return GLOBAL_PRINT_OPTIONS

    fn __exit__(mut self):
        GLOBAL_PRINT_OPTIONS.set_options(
            precision=DEFAULT_PRECISION,
            suppress_small=DEFAULT_SUPPRESS_SMALL,
            separator=DEFAULT_SEPARATOR,
            padding=DEFAULT_PADDING,
            threshold=DEFAULT_THRESHOLD,
            line_width=DEFAULT_LINE_WIDTH,
            edge_items=DEFAULT_EDGE_ITEMS,
            sign=DEFAULT_SIGN,
            float_format=DEFAULT_FLOAT_FORMAT,
            complex_format=DEFAULT_COMPLEX_FORMAT,
            nan_string=DEFAULT_NAN_STRING,
            inf_string=DEFAULT_INF_STRING,
            formatted_width=DEFAULT_FORMATTED_WIDTH,
            exponent_threshold=DEFAULT_EXPONENT_THRESHOLD,
            suppress_scientific=DEFAULT_SUPPRESS_SCIENTIFIC,
        )


fn set_printoptions(
    precision: Int = DEFAULT_PRECISION,
    suppress_small: Bool = DEFAULT_SUPPRESS_SMALL,
    separator: String = DEFAULT_SEPARATOR,
    padding: String = DEFAULT_PADDING,
    edge_items: Int = DEFAULT_EDGE_ITEMS,
):
    GLOBAL_PRINT_OPTIONS.set_options(
        precision,
        suppress_small,
        separator,
        padding,
        edge_items,
    )


# FIXME: fix the problem where precision > number of digits in the mantissa results in a not so exact value.
fn format_floating_scientific[
    dtype: DType = DType.float64
](x: Scalar[dtype], precision: Int = 10, sign: Bool = False) raises -> String:
    """
    Format a float in scientific notation.

    Parameters:
        dtype: Datatype of the float.

    Args:
        x: The float to format.
        precision: The number of decimal places to include in the mantissa.
        sign: Whether to include the sign of the float in the result. Defaults to False.

    Returns:
        A string representation of the float in scientific notation.

    Raises:
        Error: If the dtype is not a floating-point type or if precision is negative.
    """

    @parameter
    if is_inttype[dtype]():
        raise Error(
            "Invalid type provided. dtype must be a floating-point type."
        )
    if precision < 0:
        raise Error("Precision must be a non-negative integer.")

    try:
        var suppress_scientific = GLOBAL_PRINT_OPTIONS.suppress_scientific
        var exponent_threshold = GLOBAL_PRINT_OPTIONS.exponent_threshold
        var formatted_width = GLOBAL_PRINT_OPTIONS.formatted_width

        if x == 0.0:
            var result: String = "0." + "0" * precision
            return result.rjust(formatted_width)

        var power: Int = int(mt.log10(abs(x)))
        var mantissa: Scalar[dtype] = x / pow(10.0, power).cast[dtype]()
        var m_string: String = String("{0}").format(mantissa)
        var result: String = m_string[0] + "."

        for i in range(2, precision + 2):
            if i >= len(m_string):
                result += "0"
            else:
                result += m_string[i]

        if suppress_scientific and abs(power) <= exponent_threshold:
            return format_floating_precision(x, precision, sign).rjust(
                formatted_width
            )

        var exponent_str: String
        if power < 0:
            exponent_str = String("e{0}").format(power)
        else:
            exponent_str = String("e+{0}").format(power)

        if x < 0:
            return (
                String("{0}{1}")
                .format(result, exponent_str)
                .rjust(formatted_width)
            )
        if sign:
            return (
                String("+{0}{1}")
                .format(result, exponent_str)
                .rjust(formatted_width)
            )
        return (
            String("{0}{1}").format(result, exponent_str).rjust(formatted_width)
        )
    except:
        raise Error("Failed to format float in scientific notation.")


fn format_floating_precision[
    dtype: DType
](value: Scalar[dtype], precision: Int, sign: Bool = False) raises -> String:
    """
    Format a floating-point value to the specified precision.

    Args:
        value: The value to format.
        precision: The number of decimal places to include.
        sign: Whether to include the sign of the float in the result.
            Defaults to False.

    Returns:
        The formatted value as a string.

    Raises:
        Error: If precision is negative or if the value cannot be formatted.
    """

    @parameter
    if is_inttype[dtype]():
        raise Error(
            "Invalid type provided. dtype must be a floating-point type."
        )

    if precision < 0:
        raise Error("Precision must be a non-negative integer.")

    var suppress_small = GLOBAL_PRINT_OPTIONS.suppress_small
    if suppress_small and abs(value) < 1e-10:
        var result: String = String("0.")
        for _ in range(precision):
            result += "0"
        return result

    var scaling_factor = 10**precision
    var rounded_value = round(value * scaling_factor) / scaling_factor

    var integer_part = int(rounded_value)
    var fractional_part = abs(rounded_value - integer_part)

    var result = str(integer_part)
    if Scalar[dtype](0) > rounded_value > Scalar[dtype](-1):
        result = "-" + result

    if precision > 0:
        result += "."
        for _ in range(precision):
            fractional_part *= 10
            var digit = int(fractional_part)
            result += str(digit)
            fractional_part -= digit

    if sign and value > 0:
        result = "+" + result

    return result


fn format_floating_precision[
    cdtype: CDType, dtype: DType
](value: ComplexSIMD[cdtype, dtype=dtype]) raises -> String:
    """
    Format a complex floating-point value to the specified precision.

    Args:
        value: The complex value to format.

    Returns:
        The formatted value as a string.

    Raises:
        Error: If the complex value cannot be formatted.
    """
    try:
        var precision = GLOBAL_PRINT_OPTIONS.precision
        var sign = GLOBAL_PRINT_OPTIONS.sign

        return (
            "("
            + format_floating_precision(value.re, precision, sign)
            + " + "
            + format_floating_precision(value.im, precision, sign)
            + " j)"
        )
    except:
        raise Error("Failed to format complex floating-point value.")


fn format_value[
    dtype: DType
](value: Scalar[dtype], print_options: PrintOptions) raises -> String:
    """
    Format a single value based on the print options.

    Args:
        value: The value to format.
        print_options: The print options.

    Returns:
        The formatted value as a string.
    """
    var sign = print_options.sign
    var float_format = print_options.float_format
    var nan_string = print_options.nan_string
    var inf_string = print_options.inf_string
    var formatted_width = print_options.formatted_width

    @parameter
    if is_floattype[dtype]():
        if isnan(value):
            return nan_string.rjust(formatted_width)
        if isinf(value):
            return inf_string.rjust(formatted_width)
        if float_format == "scientific":
            return format_floating_scientific(
                value, print_options.precision, sign
            )
        else:
            return format_floating_precision(
                value, print_options.precision, sign
            ).rjust(formatted_width)
    else:
        var formatted = str(value)
        if sign and value > 0:
            formatted = "+" + formatted
        return formatted.rjust(formatted_width)


fn format_value[
    cdtype: CDType, dtype: DType
](
    value: ComplexSIMD[cdtype, dtype=dtype],
    print_options: PrintOptions,
) raises -> String:
    """
    Format a complex value based on the print options.

    Args:
        value: The complex value to format.
        print_options: The print options.

    Returns:
        The formatted value as a string.
    """

    var sign = print_options.sign
    var float_format = print_options.float_format
    var nan_string = print_options.nan_string
    var inf_string = print_options.inf_string
    var formatted_width = print_options.formatted_width
    var complex_format = print_options.complex_format

    var re_str: String
    var im_str: String

    if dtype.is_floating_point():
        if isnan(value.re):
            re_str = nan_string
        elif isinf(value.re):
            re_str = inf_string
        else:
            if float_format == "scientific":
                re_str = format_floating_scientific(
                    value.re, print_options.precision, sign
                )
            else:
                re_str = format_floating_precision(
                    value.re, print_options.precision, sign
                )

        if isnan(value.im):
            im_str = nan_string
        elif isinf(value.im):
            im_str = inf_string
        else:
            if float_format == "scientific":
                im_str = format_floating_scientific(
                    value.im, print_options.precision, sign
                )
            else:
                im_str = format_floating_precision(
                    value.im, print_options.precision, sign
                )

        if value.re == 0 and value.im == 0:
            im_str = "+" + im_str
    else:
        re_str = str(value.re)
        im_str = str(value.im)
        if sign:
            if value.re >= 0:
                re_str = "+" + re_str
            if value.im >= 0:
                im_str = "+" + im_str
            elif value.im <= 0:
                im_str = "-" + im_str.replace("-", "")
        else:
            if value.im <= 0:
                im_str = "-" + im_str.replace("-", "")

    re_str = re_str.rjust(formatted_width)
    im_str = im_str.rjust(formatted_width)

    if complex_format == "parentheses":
        return String("({0} {1}j)").format(re_str, im_str)
    else:
        return String("{0} {1}j").format(re_str, im_str)
