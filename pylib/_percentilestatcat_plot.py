from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import seaborn as sns


def percentilestatcat_plot(
    data: pd.DataFrame, x: str, y: str, hue: str
) -> plt.Axes:
    assert x == hue

    ax = sns.boxenplot(
        data=data,
        y=y,
        x=x,
        hue=hue,
        legend=False,
    )
    sns.barplot(
        data=data,
        y=y,
        x=x,
        hue=hue,
        alpha=0.0,
        ax=ax,
        legend=False,
    )

    max_pts = 2_000
    sampled = data.groupby(x, group_keys=False).apply(
        lambda grp: grp.sample(n=min(len(grp), max_pts), random_state=1)
    )
    sns.stripplot(
        y=sampled[y] + np.random.uniform(-1, 1, len(sampled)),
        x=sampled[x],
        alpha=0.2,
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

    for j, h_cat in enumerate(data[hue].unique().tolist()):
        grp = data[(data[hue] == h_cat)][y].dropna()
        n = len(grp)
        if n == 0:
            continue

        # --- bootstrap test for mean < threshold ---
        boot_means = np.random.choice(
            grp, size=(n_boot, n), replace=True
        ).mean(axis=1)
        p_boot = np.mean(boot_means >= threshold)

        # --- binomial/sign test for median < threshold ---
        k = np.sum(grp < threshold)
        n = np.sum(grp != threshold & ~np.isnan(grp))
        p_binom = scipy_stats.binomtest(
            k, n, p=null_p, alternative="greater"
        ).pvalue

        grp.max()
        text = (
            f"n={n}\n" f"binom<50: {p_binom:.3f}\n" f"boots<50: {p_boot:.3f}"
        )
        ax.text(j, 5, text, ha="center", va="bottom", fontsize="small")

    return ax
