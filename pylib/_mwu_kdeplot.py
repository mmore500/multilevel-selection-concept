import typing

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu
import seaborn as sns


def mwu_kdeplot(
    data: pd.DataFrame,
    x: str,
    hue: str,
    ax: typing.Optional[plt.Axes] = None,
    **kwargs: dict,
) -> plt.Axes:
    """
    Create a KDE plot with a statistical annotation in the bottom right corner.

    Parameters:
        data (pd.DataFrame): Input data.
        x (str): Column name for the x-axis variable.
        hue (str): Column name for grouping. Must have two unique values.
        ax (matplotlib.axes.Axes, optional): Axis to plot on.
        **kwargs: Other keyword arguments passed to sns.kdeplot.

    Returns:
        matplotlib.axes.Axes: The axis with the plotted KDE and annotation.
    """
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots()

    # Create the KDE plot
    sns.kdeplot(data=data, x=x, hue=hue, ax=ax, **kwargs)

    # Determine the unique groups by hue and sort them for consistency.
    hue_levels = sorted(data[hue].unique())
    palette = kwargs.get("palette", sns.color_palette())

    # Gather the data for each hue level.
    groups = [data.loc[data[hue] == level, x].dropna() for level in hue_levels]

    # Only annotate if exactly two groups exist.
    if len(groups) == 2:
        group0, group1 = groups
        # Compute medians for directionality
        median0 = group0.median()
        median1 = group1.median()

        # Perform two-sided Mann–Whitney U test.
        U, p_val = mannwhitneyu(group0, group1, alternative="two-sided")
        n0, n1 = len(group0), len(group1)
        # Compute Cliff's delta: δ = 1 - 2*(U/(n0*n1))
        cliffs_delta = 1 - 2 * (U / (n0 * n1))

        # Convert p-value to significance stars.
        if p_val < 0.00001:
            stars = "*****"
        elif p_val < 0.0001:
            stars = "****"
        elif p_val < 0.001:
            stars = "***"
        elif p_val < 0.01:
            stars = "**"
        elif p_val < 0.05:
            stars = "*"
        else:
            stars = "ns"

        # Choose the arrow direction based on the medians.
        if median0 > median1:
            arrow = "<"
        elif median0 < median1:
            arrow = ">"
        else:
            arrow = "="

        # Compose the annotation: first line significance and arrow;
        # second line Cliff's delta.
        annotation = f"{stars} {arrow}\nδ={cliffs_delta:.2f}"

        # Place the annotation into the bottom right-hand corner.
        ax.text(
            0.95,
            0.05,
            annotation,
            color={
                ">": palette[0],
                "<": palette[1],
                "=": "black",
            }[arrow],
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
        )

    return ax
