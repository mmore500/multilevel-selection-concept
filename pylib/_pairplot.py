import typing

from matplotlib.lines import Line2D as mpl_Line2D
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu  # Use Mann–Whitney U test for two groups
import seaborn as sns


def pairplot(
    data_df: pd.DataFrame,
    hue: str,
    vars_to_plot: typing.List[str],
    log_vars: typing.List[str],
) -> plt.Figure:
    """Analog of sns.Pairplot, handled manually to prevent post-hoc log scaling
    artifacts."""
    # Compute the hue levels once for consistency and for the legend
    hue_levels = sorted(data_df[hue].unique())

    n = len(vars_to_plot)
    fig, axes = plt.subplots(n, n, figsize=(2 * n, 2 * n))

    # Create custom legend handles using the default seaborn palette
    palette = sns.color_palette()

    for i, y_var in enumerate(vars_to_plot):
        for j, x_var in enumerate(vars_to_plot):
            ax = axes[i, j]

            # Diagonal: Univariate KDE plots
            if i == j:
                for level in hue_levels:
                    mask = data_df[hue] == level
                    sns.kdeplot(
                        x=data_df.loc[mask, x_var],
                        fill=True,
                        common_norm=False,
                        label=level,
                        ax=ax,
                        log_scale=(x_var in log_vars, False),
                        legend=False,  # disable individual legends
                    )

                # For a two-group scenario,
                # report the Mann–Whitney U test and Cliff's delta.
                groups = [
                    data_df.loc[data_df[hue] == level, x_var].dropna()
                    for level in hue_levels
                ]
                if len(groups) == 2:
                    group0 = groups[0]
                    group1 = groups[1]
                    # Compute medians for directionality
                    median0 = group0.median()
                    median1 = group1.median()
                    # Perform Mann–Whitney U test (two-sided)
                    U, p_val = mannwhitneyu(
                        group0, group1, alternative="two-sided"
                    )
                    n0 = len(group0)
                    n1 = len(group1)
                    # Compute Cliff's delta: δ = 1 - 2*(U/(n0*n1))
                    cliffs_delta = 1 - 2 * (U / (n0 * n1))

                    # Convert p-value to significance stars
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

                    # Determine the direction arrow:
                    #   - '>' if group0 > group1,
                    #   - '<' if group0 < group1,
                    #   - '=' if medians are exactly equal
                    if median0 > median1:
                        arrow = "<"
                    elif median0 < median1:
                        arrow = ">"
                    else:
                        arrow = "="

                    # Annotation: first line stars and arrow, second line Cliff's delta
                    annotation = f"{stars} {arrow}\nδ={cliffs_delta:.2f}"
                    ax.text(
                        0.95,
                        0.95,
                        annotation,
                        color={
                            ">": palette[0],
                            "<": palette[1],
                            "=": "black",
                        }[arrow],
                        transform=ax.transAxes,
                        horizontalalignment="right",
                        verticalalignment="top",
                    )

            # Off-diagonals: Scatterplots using seaborn's KDE plots
            else:
                # Determine log_scale parameters for x and y axes
                sns.kdeplot(
                    data=data_df,
                    x=x_var,
                    y=y_var,
                    hue=hue,
                    alpha=0.8,
                    ax=ax,
                    fill=False,
                    log_scale=(x_var in log_vars, y_var in log_vars),
                    legend=False,
                    warn_singular=False,
                )

            # Label only the left and bottom plots for clarity
            if j == 0:
                ax.set_ylabel(y_var)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            if i == n - 1:
                ax.set_xlabel(x_var)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

    # Adjust layout to make space for the top legend
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    handles = [
        mpl_Line2D([], [], color=palette[i], lw=2)
        for i, level in enumerate(hue_levels)
    ]

    # Add a flat legend along the top
    fig.legend(
        handles,
        hue_levels,
        frameon=False,
        loc="upper center",
        ncol=len(hue_levels),
        title=hue,
    )

    return fig
