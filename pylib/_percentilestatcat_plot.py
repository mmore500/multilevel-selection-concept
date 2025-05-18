import itertools as it

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import seaborn as sns


def _bootstrap_p_value(
    grp: pd.Series,
    threshold: float,
    n_boot: int = 100_000,
    max_mem_bytes: int = 4 * 1024**3,
) -> float:
    n = len(grp)

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


def percentilestatcat_plot(
    data: pd.DataFrame, x: str, y: str, hue: str
) -> plt.Axes:
    assert x == hue

    order = data[x].unique().tolist()
    data = data[~data[y].isna()].copy()

    ax = sns.boxenplot(
        data=data,
        y=y,
        x=x,
        order=order,
        hue=hue,
        hue_order=order,
        legend=False,
    )
    sns.barplot(
        data=data,
        y=y,
        x=x,
        order=order,
        hue=hue,
        hue_order=order,
        alpha=0.0,
        ax=ax,
        legend=False,
    )

    max_pts = 4_000
    sampled = (
        data.sample(frac=1, random_state=1)  # shuffle
        .reset_index(drop=True)
        .groupby(x, group_keys=False)
        .head(max_pts)
        .copy()
        .reset_index(drop=True)
    )
    sampled[y] += np.random.uniform(-1, 1, len(sampled))
    sns.stripplot(
        data=sampled,
        x=x,
        y=y,
        order=order,
        alpha=0.1,
        ax=ax,
        color="k",
        legend=False,
        jitter=0.3,
        size=4,
    )

    threshold = 50
    ax.axhline(50, color="k", linestyle="--")

    n_boot = 10_000
    null_p = (
        data.loc[
            (data[hue] == "null")
            & (data[y] != threshold)
            & ~np.isnan(data[y]),
            y,
        ]
        < threshold
    ).mean()

    for j, cat in enumerate(order):
        grp = data[(data[hue] == cat)][y].dropna()
        n = len(grp)
        if n == 0:
            continue

        # --- bootstrap test for mean < threshold ---
        p_boot = _bootstrap_p_value(grp, threshold, n_boot=n_boot)

        # --- binomial/sign test for median < threshold ---
        k = np.sum(grp < threshold)
        n = np.sum(grp != threshold & ~np.isnan(grp))
        p_binom = scipy_stats.binomtest(
            k, n, p=null_p, alternative="greater"
        ).pvalue

        grp.max()
        na = np.sum(grp.isna())
        text = (
            f"n={len(grp)} n'={n} na={na}\n"
            f"binom<50: {p_binom:.3f}\n"
            f"boots<50: {p_boot:.3f}"
        )
        ax.text(j, 5, text, ha="center", va="bottom", fontsize="small")

    return ax
