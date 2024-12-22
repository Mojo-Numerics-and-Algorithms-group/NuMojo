"""
Kernal Density Estimation
"""
from math import exp, cos

from ...core.ndarray import NDArray
from ..math import exp as aexp, cos as Acos, sum
from ...logic.comparison import less
from ..constants import Constants

alias pi = Constants().pi


trait DensityKernal:
    @staticmethod
    fn kernal[
        dtype: DType
    ](x: Scalar[dtype], h: Scalar[dtype]) -> Scalar[dtype]:
        pass

    @staticmethod
    fn kernal[
        dtype: DType
    ](x: NDArray[dtype], h: Scalar[dtype]) raises -> NDArray[dtype]:
        pass


struct Guassian(DensityKernal):
    @staticmethod
    fn kernal[
        dtype: DType
    ](x: Scalar[dtype], h: Scalar[dtype]) -> Scalar[dtype]:
        return exp(-((x**2) / (2 * (h**2))))

    @staticmethod
    fn kernal[
        dtype: DType
    ](x: NDArray[dtype], h: Scalar[dtype]) raises -> NDArray[dtype]:
        return aexp(-((x**2) / (2 * (h**2))))


struct TopHat(DensityKernal):
    @staticmethod
    fn kernal[
        dtype: DType
    ](x: Scalar[dtype], h: Scalar[dtype]) -> Scalar[dtype]:
        return Scalar[dtype]((x < h).cast[dtype]())

    @staticmethod
    fn kernal[
        dtype: DType
    ](x: NDArray[dtype], h: Scalar[dtype]) raises -> NDArray[dtype]:
        return less(x, h).astype[dtype]()


struct Epanechnikov(DensityKernal):
    @staticmethod
    fn kernal[
        dtype: DType
    ](x: Scalar[dtype], h: Scalar[dtype]) -> Scalar[dtype]:
        return 1 - exp(((x**2) / ((h**2))))

    @staticmethod
    fn kernal[
        dtype: DType
    ](x: NDArray[dtype], h: Scalar[dtype]) raises -> NDArray[dtype]:
        return 1 - aexp(((x**2) / ((h**2))))


struct Exponential(DensityKernal):
    @staticmethod
    fn kernal[
        dtype: DType
    ](x: Scalar[dtype], h: Scalar[dtype]) -> Scalar[dtype]:
        return exp(-x / h)

    @staticmethod
    fn kernal[
        dtype: DType
    ](x: NDArray[dtype], h: Scalar[dtype]) raises -> NDArray[dtype]:
        return aexp(-x / h)


struct Linear(DensityKernal):
    @staticmethod
    fn kernal[
        dtype: DType
    ](x: Scalar[dtype], h: Scalar[dtype]) -> Scalar[dtype]:
        return Scalar[dtype]((x < h).cast[dtype]()) * ((1 - x) / h)

    @staticmethod
    fn kernal[
        dtype: DType
    ](x: NDArray[dtype], h: Scalar[dtype]) raises -> NDArray[dtype]:
        return less(x, h).astype[dtype]() * ((1 - x) / h)


struct Cosine(DensityKernal):
    @staticmethod
    fn kernal[
        dtype: DType
    ](x: Scalar[dtype], h: Scalar[dtype]) -> Scalar[dtype]:
        return Scalar[dtype]((x < h).cast[dtype]()) * cos(pi * x / (2 * h))

    @staticmethod
    fn kernal[
        dtype: DType
    ](x: NDArray[dtype], h: Scalar[dtype]) raises -> NDArray[dtype]:
        return less(x, h).astype[dtype]() * Acos(pi * x / (2 * h))


# TODO make this able to take multiple dimension arrays
struct KDE[kernal: DensityKernal]:
    """
    Performs Kernal Density Estimations.
    """
    @staticmethod
    fn eval[
        dtype: DType
    ](
        x: NDArray[dtype], y: Scalar[dtype], h: Scalar[dtype]
    ) raises -> Scalar[dtype]:
        """
        Evaluate the KDE at a single point.
        Args:
            x: The data for which the density estimation is being done.
            y: The point of Evaluation.

        Returns:
            The value of the probability distrobution at y.
        """
        if x.shape.len() > 1:
            raise Error(
                "Dimensions greater than one not currently allowed for KDE"
            )

        return sum(kernal.kernal(y - x, h))

    @staticmethod
    fn eval[
        dtype: DType
    ](
        x: NDArray[dtype], y: NDArray[dtype], h: Scalar[dtype]
    ) raises -> NDArray[dtype]:
        """
        Evaluate the KDE at a single point.
        Args:
            x: The data for which the density estimation is being done.
            y: The points of Evaluation.

        Returns:
            The values of the probability distrobution at each value in y.
        """
        if x.shape.len() > 1 or y.shape.len() > 1:
            raise Error(
                "Dimensions greater than one not currently allowed for KDE"
            )
        var res: NDArray[dtype] = NDArray[dtype](
            VariadicList[Int](y.shape[0], x.shape[0])
        )
        print(res[:, 0 : 0 + 1].shape)
        for i in range(x.shape[0]):
            res[:, i : i + 1] = y - x[i]
        return sum(kernal.kernal(res, h), axis=1)
