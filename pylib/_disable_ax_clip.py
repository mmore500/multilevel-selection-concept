from matplotlib import pyplot as plt


def disable_ax_clip(ax: plt.Axes) -> None:
    ax.set_clip_on(False)
    for artist in (
        ax.artists + ax.patches + ax.lines + ax.texts + ax.collections
    ):
        artist.set_clip_on(False)
