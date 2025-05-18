import typing

import numpy as np


def bootstrap_extrema_values(
    data: np.ndarray,
    sample_size: int,
    weights: typing.Optional[np.ndarray] = None,
    extrema: typing.Callable = np.max,
    n_bootstrap: int = 1000,
) -> float:
    """Sample a bootstrap distribution of extrema values."""

    # Convert data to a numpy array
    data = np.asarray(data)

    if weights is not None and np.sum(weights) > 0:
        weights = weights.copy()
        weights[np.isnan(weights)] = 0.0
        p = weights / np.sum(weights)
    else:
        p = None

    samples = np.random.choice(
        data,
        p=p,
        size=(n_bootstrap, sample_size),
        replace=True,
    )
    extrema_values = np.apply_along_axis(extrema, 1, samples)
    return extrema_values
