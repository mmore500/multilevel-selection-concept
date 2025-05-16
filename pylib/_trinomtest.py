import numbers
import typing

import numpy as np
from scipy import stats as scipy_stats


def trinomtest_fast(
    data: typing.Sequence[numbers.Real],
    mu: numbers.Real = 0,
    nan_policy: typing.Literal["propagate", "omit", "raise"] = "propagate",
) -> float:
    """Calculate the two-tailed p-value for a trinomial test, under the null
    hypothesis that the data have median mu.

    Extends the binomial test to take into account the number of ties in the data.

    References
    ----------
    Bian, Guorui and McAleer, Michael and Wong, Wing-Keung, A Trinomial Test
    for Paired Data When There are Many Ties (September 8, 2009).
    http://dx.doi.org/10.2139/ssrn.1410589

    https://peterstatistics.com/Packages/python-docs/stikpetP/tests/test_trinomial_os.html
    """
    data = np.asarray(data) - mu
    if nan_policy == "omit":
        data = data[~np.isnan(data)]  # remove NaNs
    elif np.isnan(data).any():
        if nan_policy == "raise":
            raise ValueError("NaN values found in data")
        elif nan_policy == "propagate":
            return np.nan

    n = len(data)
    nd = abs(np.sign(data).astype(int).sum())
    pTie = np.mean(data == 0)

    t = np.arange(0, n + 1)  # possible tie‐counts
    pmf_t = scipy_stats.binom.pmf(t, n, pTie)  # P{T=t}
    m = n - t
    # for each t, threshold on C = ceil((m+nd)/2)
    thresh = np.ceil((m + nd) / 2).astype(int)
    tail = scipy_stats.binom.sf(thresh - 1, m, 0.5)  # P{C >= thresh}
    res = np.dot(pmf_t, tail)

    assert np.isnan(res) or 0.0 <= res <= 1.0
    return res


def trinomtest_naive(
    data: typing.Sequence[numbers.Real],
    mu: numbers.Real = 0,
    nan_policy: typing.Literal["propagate", "omit", "raise"] = "propagate",
) -> float:
    """Calculate the two-tailed p-value for a trinomial test, under the null
    hypothesis that the data have median mu.

    Extends the binomial test to take into account the number of ties in the data.

    References
    ----------
    Bian, Guorui and McAleer, Michael and Wong, Wing-Keung, A Trinomial Test
    for Paired Data When There are Many Ties (September 8, 2009).
    http://dx.doi.org/10.2139/ssrn.1410589

    https://peterstatistics.com/Packages/python-docs/stikpetP/tests/test_trinomial_os.html
    """
    data = np.asarray(data) - mu

    if nan_policy == "omit":
        data = data[~np.isnan(data)]  # remove NaNs
    elif np.isnan(data).any():
        if nan_policy == "raise":
            raise ValueError("NaN values found in data")
        elif nan_policy == "propagate":
            return np.nan

    n = len(data)
    nd = abs(np.sign(data).astype(int).sum())
    pTie = np.mean(data == 0)

    summands = (
        scipy_stats.multinomial.pmf(
            x=(k, k + d, n - 2 * k - d),
            n=n,
            p=((1.0 - pTie) / 2, (1.0 - pTie) / 2, pTie),
        )
        for d in range(nd, n + 1)
        for k in range(((n - d) // 2) + 1)
    )

    res = sum(summands)
    assert np.isnan(res) or 0.0 <= res <= 1.0
    return res
