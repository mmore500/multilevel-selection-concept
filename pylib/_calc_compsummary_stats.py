import itertools as it

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats


def _bootstrap_p_value(
    grp: pd.Series,
    threshold: float,
    n_boot: int = 100_000,
    max_mem_bytes: int = 1024**3,
) -> float:
    n = len(grp)

    if n == 0:
        return np.nan

    bytes_per_resample = n * grp.dtype.itemsize
    batch_size = max(1, min(n_boot, int(max_mem_bytes // bytes_per_resample)))

    successes = 0
    for begin, end in it.pairwise(
        range(0, n_boot + batch_size - 1, batch_size)
    ):
        end = min(end, n_boot)
        means = np.random.choice(
            grp, size=(end - begin, n), replace=True
        ).mean(axis=1)
        successes += np.sum(means >= threshold)

    return successes / n_boot


def calc_compsummary_stats(data: pd.DataFrame, x: str, y: str) -> plt.Axes:

    data = data.dropna(subset=y)

    threshold = 50

    n_boot = 10_000
    null_p = (data.loc[(data[y] != threshold), y] < threshold).mean()
    if np.isnan(null_p):
        null_p = 0.5

    results = []
    for label, grp in {
        "all": data[y],
        **{k: v[y] for k, v in data.groupby(x, observed=False)},
    }.items():

        # --- bootstrap test for mean < threshold ---
        p_boot = _bootstrap_p_value(grp, threshold, n_boot=n_boot)

        # --- binomial/sign test for median < threshold ---
        binom_k = np.sum(grp < threshold)
        binom_n = np.sum(grp != threshold & ~np.isnan(grp))
        try:
            p_binom = scipy_stats.binomtest(
                binom_k, binom_n, p=null_p, alternative="greater"
            ).pvalue
        except ValueError:
            p_binom = np.nan

        # --- mann-whitney test for difference in medians ---
        if label != "all":
            focal = data.loc[data[x] == label, y].dropna()
            nonfocal = data.loc[data[x] != label, y].dropna()
            mw_u, mw_p = scipy_stats.mannwhitneyu(
                focal, nonfocal, alternative="greater"
            )
            cliffs_delta = 2 * mw_u / (len(focal) * len(nonfocal)) - 1
        else:
            mw_u, mw_p, cliffs_delta = np.nan, np.nan, np.nan

        results.append(
            {
                "label": label,
                "binom_n": binom_n,
                "binom_k": binom_k,
                "n": len(data),
                "n_boot": n_boot,
                "p_boot": p_boot,
                "p_binom": p_binom,
                "p_mw": mw_p,
                "u_mw": mw_u,
                "cliffs_delta": cliffs_delta,
                "na": np.sum(grp.isna()),
                "n_grp": len(grp),
            },
        )

    return results
