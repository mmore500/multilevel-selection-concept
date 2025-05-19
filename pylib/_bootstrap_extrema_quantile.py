import typing

import numpy as np
from scipy import stats as scipy_stats

from ._bootstrap_extrema_values import bootstrap_extrema_values


def bootstrap_extrema_quantile(
    data: np.ndarray,
    sample_value: float,
    sample_size: int,
    weights: typing.Optional[np.ndarray] = None,
    extrema: typing.Callable = np.max,
    n_bootstrap: int = 1000,
) -> float:
    """Calculate the quantile of a sample value in a bootstrap distribution of
    extrema values."""

    extrema_values = bootstrap_extrema_values(
        data,
        sample_size=sample_size,
        weights=weights,
        extrema=extrema,
        n_bootstrap=n_bootstrap,
    )
    return (
        scipy_stats.percentileofscore(
            extrema_values, sample_value, kind="rank"
        )
        / 100.0
    )
