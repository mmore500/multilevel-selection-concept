import typing

import numpy as np
from scipy import stats as scipy_stats


def bootstrap_extrema_quantile(
    data: np.ndarray,
    sample_value: float,
    sample_size: int,
    extrema: typing.Callable = np.max,
    n_bootstrap: int = 1000,
) -> float:
    """Calculate the quantile of a sample value in a bootstrap distribution of
    extrema values."""

    # Convert data to a numpy array
    data = np.asarray(data)

    samples = np.random.choice(
        data, size=(n_bootstrap, sample_size), replace=True
    )
    extrema_values = np.apply_along_axis(extrema, 1, samples)
    return (
        scipy_stats.percentileofscore(
            extrema_values, sample_value, kind="rank"
        )
        / 100.0
    )
