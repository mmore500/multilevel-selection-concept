import functools
import warnings

import numpy as np


def _ignore_errors(func):
    """
    Decorator to wrap a numpy nan* function so that it ignores
    invalid/divide warnings (e.g. all-NaN slices, division by zero).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return func(*args, **kwargs)

    return wrapper


# Wrapped versions
nanmean = _ignore_errors(np.nanmean)
nanmedian = _ignore_errors(np.nanmedian)
nanstd = _ignore_errors(np.nanstd)
nanvar = _ignore_errors(np.nanvar)
nansum = _ignore_errors(np.nansum)
nanmin = _ignore_errors(np.nanmin)
nanmax = _ignore_errors(np.nanmax)
nanprod = _ignore_errors(np.nanprod)
