"""Drawing uniforms, correlated or not, for a set of parameters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.errors import ParameterError

from .gaussian import norm_cdf, unit_interval

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["correlation_matrix", "uniforms"]

type Correlations = Mapping[tuple[str, str], float]


def correlation_matrix(labels: Sequence[str], pairs: Correlations) -> npt.NDArray[np.float64]:
    """A correlation matrix over ``labels`` with ``pairs`` filled in.

    Raises:
        ParameterError: if a pair names an unknown label or repeats one.
        ValueError: if a coefficient lies outside ``[-1, 1]``.
    """
    position = {label: index for index, label in enumerate(labels)}
    matrix = np.eye(len(labels), dtype=np.float64)
    for (first, second), coefficient in pairs.items():
        unknown = sorted({first, second} - set(position))
        if unknown:
            raise ParameterError(
                f"the correlation names parameters this model does not have: {unknown}.  "
                f"It offers {tuple(labels)}."
            )
        if first == second:
            raise ParameterError(f"{first!r} is perfectly correlated with itself by definition")
        if not -1.0 <= coefficient <= 1.0:
            raise ValueError(
                f"a correlation coefficient lies between -1 and 1, got {coefficient!r} "
                f"for {first!r} and {second!r}"
            )
        row, column = position[first], position[second]
        matrix[row, column] = matrix[column, row] = coefficient
    return matrix


def uniforms(
    rng: np.random.Generator,
    n: int,
    labels: Sequence[str],
    correlation: Correlations | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    """``n`` uniform draws per label, correlated through a Gaussian copula.

    Each marginal stays uniform, so every distribution keeps its own shape
    whatever correlation is imposed.

    Raises:
        ParameterError: if the requested correlations are not consistent.
        ValueError: if ``n`` is not positive.
    """
    if n <= 0:
        raise ValueError(f"a simulation needs at least one trial, got {n}")
    if not correlation:
        return {label: unit_interval(rng.random(n)) for label in labels}

    matrix = correlation_matrix(labels, correlation)
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as impossible:
        raise ParameterError(
            "these correlations cannot all hold at once: the matrix they describe is not "
            "positive definite.  Relax one of the coefficients, or drop a pair."
        ) from impossible
    normals = rng.standard_normal((n, len(labels))) @ factor.T
    return {label: unit_interval(norm_cdf(normals[:, index])) for index, label in enumerate(labels)}
