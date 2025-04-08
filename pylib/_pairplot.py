import typing

from matplotlib.lines import Line2D as mpl_Line2D
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kruskal
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

    for i, y_var in enumerate(vars_to_plot):
        for j, x_var in enumerate(vars_to_plot):
            ax = axes[i, j]

            # Diagonal: Univariate KDE plots
            if i == j:
                for level in hue_levels:
                    mask = data_df[hue] == level
                    if x_var in log_vars:
                        sns.kdeplot(
                            x=data_df.loc[mask, x_var],
                            fill=True,
                            common_norm=False,
                            label=level,
                            ax=ax,
                            log_scale=(True, False),
                            legend=False,  # disable individual legends
                        )
                    else:
                        sns.kdeplot(
                            x=data_df.loc[mask, x_var],
                            fill=True,
                            common_norm=False,
                            label=level,
                            ax=ax,
                            legend=False,  # disable individual legends
                        )
                # Compute the Kruskal-Wallis test across groups for the current variable
                groups = [
                    data_df.loc[data_df[hue] == level, x_var].dropna()
                    for level in hue_levels
                ]
                if len(groups) > 1:
                    stat, p_val = kruskal(*groups)
                    if p_val < 0.001:
                        sig = "***"
                    elif p_val < 0.01:
                        sig = "**"
                    elif p_val < 0.05:
                        sig = "*"
                    else:
                        sig = "ns"
                    # Annotate significance level
                    ax.text(
                        0.95,
                        0.95,
                        sig,
                        transform=ax.transAxes,
                        horizontalalignment="right",
                        verticalalignment="top",
                    )

            # Off-diagonals: Scatterplots using seaborn's KDE plots
            else:
                # Determine log_scale parameters for x and y axes
                x_log = x_var in log_vars
                y_log = y_var in log_vars
                sns.kdeplot(
                    data=data_df,
                    x=x_var,
                    y=y_var,
                    hue=hue,
                    alpha=0.8,
                    ax=ax,
                    fill=False,
                    log_scale=(x_log, y_log),
                    legend=False,
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

    # Create custom legend handles using the default seaborn palette
    palette = sns.color_palette()
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
