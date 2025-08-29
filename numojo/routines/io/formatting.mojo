from builtin.math import pow
import math as mt
from utils.numerics import isnan, isinf

from numojo.core.utility import is_inttype, is_floattype

alias DEFAULT_PRECISION = 4
alias DEFAULT_SUPPRESS_SMALL = False
alias DEFAULT_SEPARATOR = " "
alias DEFAULT_PADDING = ""
alias DEFAULT_EDGE_ITEMS = 2
alias DEFAULT_THRESHOLD = 15
alias DEFAULT_LINE_WIDTH = 75
alias DEFAULT_SIGN = False
alias DEFAULT_FLOAT_FORMAT = "fixed"
alias DEFAULT_COMPLEX_FORMAT = "parentheses"
alias DEFAULT_NAN_STRING = "nan"
alias DEFAULT_INF_STRING = "inf"
alias DEFAULT_FORMATTED_WIDTH = 6
alias DEFAULT_EXPONENT_THRESHOLD = 4
alias DEFAULT_SUPPRESS_SCIENTIFIC = False

# placeholder, we can use this global var option in future when Mojo supports global options
alias GLOBAL_PRINT_OPTIONS = PrintOptions(
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


struct PrintOptions(Copyable, Movable):
    var precision: Int
    """
    The number of decimal places to include in the formatted string.
    Defaults to 4.
    """
    var suppress_small: Bool
    var separator: String
    """
    The separator between elements in the array. Defaults to a space.
    """
    var padding: String
    """
    The padding symbol between the elements at the edge and the brackets.
    Defaults to an empty string.
    """
    var threshold: Int
    var line_width: Int
    var edge_items: Int
    """
    The number of items to display at the beginning and end of a dimension.
    Defaults to 3.
    """
    var sign: Bool
    var float_format: String
    var complex_format: String
    var nan_string: String
    var inf_string: String
    var formatted_width: Int
    """
    The width of the formatted string per element of array.
    """
    var exponent_threshold: Int
    var suppress_scientific: Bool

    fn __init__(
        out self,
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
        var default_print_options = PrintOptions()
        default_print_options.set_options(
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
        return default_print_options

    fn __exit__(mut self):
        var default_print_options = PrintOptions()
        default_print_options.set_options(
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
    var default_print_options = PrintOptions()
    default_print_options.set_options(
        precision,
        suppress_small,
        separator,
        padding,
        edge_items,
    )


# FIXME: fix the problem where precision > number of digits in the mantissa results in a not so exact value.
fn format_floating_scientific[
    dtype: DType = DType.float64
](
    x: Scalar[dtype],
    precision: Int = 10,
    sign: Bool = False,
    suppress_scientific: Bool = False,
    exponent_threshold: Int = 4,
    formatted_width: Int = 8,
) raises -> String:
    """
    Format a float in scientific notation.

    Notes: A scientific notation takes the form `-a.bbbbe+ii`. It will take
    `7 + precision` letters in total.

    Parameters:
        dtype: Datatype of the float.

    Args:
        x: The float to format.
        precision: The number of decimal places to include in the mantissa.
        sign: Whether to include the sign of the float in the result. Defaults to False.
        suppress_scientific: Whether to suppress scientific notation for small numbers.
            Defaults to False.
        exponent_threshold: The threshold for suppressing scientific notation.
            Defaults to 4.
        formatted_width: The width of the formatted string. Defaults to 8.

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
        if x == 0:
            if sign:
                var result: String = "+0." + "0" * precision + "e+00"
                return result.rjust(formatted_width)
            else:
                var result: String = " 0." + "0" * precision + "e+00"
                return result.rjust(formatted_width)

        var power: Int = Int(mt.log10(abs(x)))
        if Scalar[dtype](0.0) < abs(x) < Scalar[dtype](1.0):
            power -= 1
        var mantissa: Scalar[dtype] = x / pow(10.0, power).cast[dtype]()
        var mantissa_without_sign_string = String(abs(mantissa))

        var result: String
        if x < 0:
            result = "-" + mantissa_without_sign_string[: 2 + precision]
        else:
            if sign:
                result = "+" + mantissa_without_sign_string[: 2 + precision]
            else:
                result = " " + mantissa_without_sign_string[: 2 + precision]

        if suppress_scientific and abs(power) <= exponent_threshold:
            return format_floating_precision(x, precision, sign).rjust(
                formatted_width
            )

        var exponent_string: String
        if power < 0:
            if power > -10:
                exponent_string = String("e-0{0}").format(-power)
            else:
                exponent_string = String("e-{0}").format(-power)
        else:
            if power < 10:
                exponent_string = String("e+0{0}").format(power)
            else:
                exponent_string = String("e+{0}").format(power)

        return (
            String("{0}{1}")
            .format(result, exponent_string)
            .rjust(formatted_width)
        )
    except:
        raise Error("Failed to format float in scientific notation.")


fn format_floating_precision[
    dtype: DType
](
    value: Scalar[dtype],
    precision: Int,
    sign: Bool = False,
    suppress_small: Bool = False,
) raises -> String:
    """
    Format a floating-point value to the specified precision.

    Args:
        value: The value to format.
        precision: The number of decimal places to include.
        sign: Whether to include the sign of the float in the result.
            Defaults to False.
        suppress_small: Whether to suppress small numbers. Defaults to False.

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

    if suppress_small and abs(value) < 1e-10:
        var result: String = String("0.")
        for _ in range(precision):
            result += "0"
        return result

    var scaling_factor = 10**precision
    var rounded_value = round(value * scaling_factor) / scaling_factor

    var integer_part = Int(rounded_value)
    var fractional_part = abs(rounded_value - integer_part)

    var result = String(integer_part)
    if Scalar[dtype](0) > rounded_value > Scalar[dtype](-1):
        result = "-" + result

    if precision > 0:
        result += "."
        for _ in range(precision):
            fractional_part *= 10
            var digit = Int(fractional_part)
            result += String(digit)
            fractional_part -= digit

    if sign and value > 0:
        result = "+" + result

    return result


fn format_floating_precision[
    dtype: DType
](
    value: ComplexSIMD[dtype],
    precision: Int = 4,
    sign: Bool = False,
) raises -> String:
    """
    Format a complex floating-point value to the specified precision.

    Args:
        value: The complex value to format.
        precision: The number of decimal places to include.
        sign: Whether to include the sign of the float in the result.
            Defaults to False.

    Returns:
        The formatted value as a string.

    Raises:
        Error: If the complex value cannot be formatted.
    """
    try:
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
    var suppress_small = print_options.suppress_small
    var suppress_scientific = print_options.suppress_scientific
    var exponent_threshold = print_options.exponent_threshold

    @parameter
    if is_floattype[dtype]():
        if isnan(value):
            return nan_string.rjust(formatted_width)
        if isinf(value):
            return inf_string.rjust(formatted_width)
        if float_format == "scientific":
            return format_floating_scientific(
                value,
                print_options.precision,
                sign,
                suppress_scientific,
                exponent_threshold,
                formatted_width,
            )
        else:
            return format_floating_precision(
                value,
                print_options.precision,
                sign,
                suppress_small,
            ).rjust(formatted_width)
    else:
        var formatted = String(value)
        if sign and value > 0:
            formatted = "+" + formatted
        return formatted.rjust(formatted_width)


fn format_value[
    dtype: DType
](value: ComplexSIMD[dtype], print_options: PrintOptions,) raises -> String:
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
    var suppress_small = print_options.suppress_small
    var suppress_scientific = print_options.suppress_scientific
    var exponent_threshold = print_options.exponent_threshold

    var re_str: String
    if dtype.is_floating_point():
        if isnan(value.re):
            re_str = nan_string
        elif isinf(value.re):
            re_str = inf_string
        else:
            if float_format == "scientific":
                re_str = format_floating_scientific(
                    value.re,
                    print_options.precision,
                    sign,
                    suppress_scientific,
                    exponent_threshold,
                    formatted_width,
                )
            else:
                re_str = format_floating_precision(
                    value.re,
                    print_options.precision,
                    sign,
                    suppress_small,
                )
    else:
        re_str = String(value.re)
        if sign and value.re >= 0:
            re_str = "+" + re_str

    # Decide sign for imaginary component and format magnitude
    var imag_sign_char: String = "+"
    var imag_mag_str: String
    if dtype.is_floating_point():
        if isnan(value.im):
            imag_mag_str = nan_string
            imag_sign_char = "+"
        elif isinf(value.im):
            if value.im < 0:
                imag_sign_char = "-"
            imag_mag_str = inf_string
        else:
            if value.im < 0:
                imag_sign_char = "-"
            var abs_im = value.im
            if abs_im < 0:
                abs_im = -abs_im
            if float_format == "scientific":
                imag_mag_str = format_floating_scientific(
                    abs_im,
                    print_options.precision,
                    False,
                    suppress_scientific,
                    exponent_threshold,
                    formatted_width,
                )
            else:
                imag_mag_str = format_floating_precision(
                    abs_im,
                    print_options.precision,
                    False,
                    suppress_small,
                )
    else:
        if value.im < 0:
            imag_sign_char = "-"
        var abs_im_int = value.im
        if abs_im_int < 0:
            abs_im_int = -abs_im_int
        imag_mag_str = String(abs_im_int)

    # Right justify parts
    re_str = re_str.rjust(formatted_width)
    imag_mag_str = imag_mag_str.rjust(formatted_width)

    return _trim_paranthesis_strings_cnumbers(
        complex_format, re_str, imag_mag_str, imag_sign_char
    )


fn _trim_paranthesis_strings_cnumbers(
    complex_format: String,
    re_str: String,
    imag_mag_str: String,
    imag_sign_char: String,
) raises -> String:
    # (a+bj) / (a-bj)
    var trim_re: String = String("")
    var seen: Bool = False
    for ch in re_str.codepoint_slices():
        if (not seen) and ch == String(" "):
            continue
        seen = True
        trim_re += ch
    var trim_im: String = String("")
    seen = False
    for ch in imag_mag_str.codepoint_slices():
        if (not seen) and ch == String(" "):
            continue
        seen = True
        trim_im += ch

    if complex_format == "parentheses":
        return String("({0} {1} {2}j)").format(trim_re, imag_sign_char, trim_im)

    return String("{0} {1} {2}j").format(trim_re, imag_sign_char, trim_im)
