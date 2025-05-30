# Copyright (C) 2009 by Eric Talevich (eric.talevich@gmail.com)
#
# This file is governed by your choice of the "Biopython License Agreement"
# or the "BSD 3-Clause License". Please see the LICENSE file at
# https://github.com/biopython/biopython/blob/137dca3eb9023fbc44ca376ae1d843038d6487af/LICENSE.rst

"""Utilities for handling, displaying and exporting Phylo trees.

Third-party libraries are loaded when the corresponding function is called.
"""

from Bio import MissingPythonDependencyError
import matplotlib
import pandas as pd


def _calc_layout(tree, y_scale=1, interp_unit=1):
    """Compute a per-clade layout DataFrame from a Biopython tree.

    Returns a DataFrame with columns:
      - id              : unique integer for each clade (row order)
      - ancestor_id     : id of its parent clade (self for root)
      - layout_position : normalized x-position (0-1 along branch-length axis)
      - y_position      : normalized y-position (0-1 among taxa)
      - color           : branch color (hex)
      - lw              : line-width in matplotlib units
    """

    def get_x_positions(tree):
        depths = tree.depths()
        if not max(depths.values()):
            depths = tree.depths(unit_branch_lengths=True)
        return depths

    def get_y_positions(tree):
        maxheight = tree.count_terminals()
        heights = {
            tip: maxheight - i
            for i, tip in enumerate(reversed(tree.get_terminals()))
        }

        def calc_row(clade):
            for sub in clade:
                if sub not in heights:
                    calc_row(sub)
            heights[clade] = (
                heights[clade.clades[0]] + heights[clade.clades[-1]]
            ) / 2.0

        if tree.root.clades:
            calc_row(tree.root)
        min_, max_ = min(heights.values()), max(heights.values())
        return {
            k: y_scale * (v - min_) / (max_ - min_) for k, v in heights.items()
        }

    x_posns = get_x_positions(tree)
    y_posns = get_y_positions(tree)
    max_x = max(x_posns.values()) or 1.0
    base_lw = matplotlib.rcParams["lines.linewidth"]

    def traverse(clade, parent=None):
        yield clade, parent
        for child in clade.clades:
            yield from traverse(child, clade)

    pairs = list(traverse(tree.root))
    id_map = {clade: idx for idx, (clade, _) in enumerate(pairs)}

    rows = []
    for clade, parent in pairs:
        cid = id_map[clade]
        pid = id_map[parent] if parent is not None else cid
        x_raw = x_posns[clade]
        color = "k"
        if hasattr(clade, "color") and clade.color is not None:
            color = clade.color.to_hex()
        lw = base_lw
        if hasattr(clade, "width") and clade.width is not None:
            lw = clade.width * base_lw

        rows.append(
            {
                "id": cid,
                "ancestor_id": pid,
                "layout_position": x_raw / max_x,
                "y_position": y_posns[clade],
                "color": color,
                "lw": lw,
            }
        )

    return pd.DataFrame(rows)


def _render_layout(
    df, do_show=True, axes=None, polar=False, interp_unit=1, **kwargs
):
    """Plot a tree from its layout DataFrame (as returned by _calc_layout),
    breaking each branch into sub-segments of max-length interp_unit."""
    import matplotlib.pyplot as plt
    import numpy as np

    # helper to split [a,b] into <=interp_unit pieces
    def _subsegs(a, b):
        total = b - a
        n = max(1, int(abs(total) // interp_unit))
        pts = np.linspace(a, b, n + 1)
        return zip(pts[:-1], pts[1:])

    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111, polar=polar)
    elif not hasattr(axes, "vlines"):
        raise ValueError(f"Invalid axes: {axes}")

    # draw horizontal (branch‐length) segments with interpolation
    for _, row in df.iterrows():
        cid = row["id"]
        pid = row["ancestor_id"]
        if pid == cid:
            continue
        parent = df.loc[df["id"] == pid].squeeze()
        y = row["y_position"]
        for xs, xe in _subsegs(
            parent["layout_position"], row["layout_position"]
        ):
            axes.vlines(
                y,
                xs,
                xe,
                color=row["color"],
                lw=row["lw"],
                capstyle="round",
                joinstyle="round",
            )

    # draw vertical (clade‐connector) segments with interpolation
    for pid, group in df.groupby("ancestor_id"):
        if pid is None or len(group) < 2:
            continue
        x_here = float(df.loc[df["id"] == pid, "layout_position"])
        color = df.loc[df["id"] == pid, "color"].iloc[0]
        lw = df.loc[df["id"] == pid, "lw"].iloc[0]
        y_bot, y_top = group["y_position"].min(), group["y_position"].max()
        for y0, y1 in _subsegs(y_bot, y_top):
            axes.hlines(
                x_here,
                y0,
                y1,
                color=color,
                lw=lw,
                capstyle="round",
                joinstyle="round",
            )

    # Aesthetics (copying original)
    axes.set_xlabel("branch length")
    axes.set_ylabel("taxa")

    xmax = df["layout_position"].max()
    axes.set_ylim(-0.05 * xmax, 1.25 * xmax)
    axes.set_xlim(df["y_position"].max() + 0.8, 0.2)

    # any extra kwargs go straight to plt
    for key, value in kwargs.items():
        try:
            list(value)
        except TypeError:
            raise ValueError(
                f'Keyword arg "{key}={value}" '
                "is not in format pyplot_option=(tuple) or (tuple,dict) or dict"
            ) from None
        if isinstance(value, dict):
            getattr(plt, key)(**value)
        elif not isinstance(value[0], tuple):
            getattr(plt, key)(*value)
        else:
            getattr(plt, key)(*value[0], **value[1])

    if do_show:
        plt.show()
    return axes


def draw_biopyhlo_vertical(
    tree,
    label_func=str,
    do_show=True,
    show_confidence=True,
    # For power users
    axes=None,
    branch_labels=None,
    label_colors=None,
    polar=False,
    y_scale=1,
    interp_unit=1,
    *args,
    **kwargs,
):
    """Plot the given tree using matplotlib (or pylab).
    (docstring unchanged)
    """
    try:
        pass
    except ImportError:
        try:
            pass
        except ImportError:
            raise MissingPythonDependencyError(
                "Install matplotlib or pylab if you want to use draw."
            ) from None

    def conf2str(conf):
        if int(conf) == conf:
            return str(int(conf))
        return str(conf)

    # branch_labels → format_branch_label (verbatim)
    if not branch_labels:
        if show_confidence:

            def format_branch_label(clade):
                try:
                    confidences = clade.confidences
                except AttributeError:
                    pass
                else:
                    return "/".join(conf2str(c.value) for c in confidences)
                if clade.confidence is not None:
                    return conf2str(clade.confidence)
                return None

        else:

            def format_branch_label(clade):
                return None

    elif isinstance(branch_labels, dict):

        def format_branch_label(clade):
            return branch_labels.get(clade)

    else:
        if not callable(branch_labels):
            raise TypeError(
                "branch_labels must be either a dict or a callable (function)"
            )
        format_branch_label = branch_labels

    # label_colors → get_label_color (verbatim)
    if label_colors:
        if callable(label_colors):

            def get_label_color(label):
                return label_colors(label)

        else:

            def get_label_color(label):
                return label_colors.get(label, "black")

    else:

        def get_label_color(label):
            return "black"

    # 1) compute layout
    layout_df = _calc_layout(tree, y_scale=y_scale, interp_unit=interp_unit)

    # 2) render with interpolation
    axes = _render_layout(
        layout_df,
        do_show=do_show,
        axes=axes,
        polar=polar,
        interp_unit=interp_unit,
        **kwargs,
    )

    # 3) overlay text labels exactly as before
    for clade in tree.find_clades():
        # map clade → df row via the id_map we generated earlier
        row = layout_df.iloc[
            _calc_layout(tree, y_scale, interp_unit)
            .query("id == @layout_df.id")
            .index[0]
        ]
        x, y = row["layout_position"], row["y_position"]

        label = label_func(clade)
        if label and label not in (None, clade.__class__.__name__):
            axes.text(
                y,
                x,
                f" {label}",
                verticalalignment="center",
                color=get_label_color(label),
            )
        conf_label = format_branch_label(clade)
        if conf_label:
            axes.text(
                y,
                0.5 * (row["layout_position"] + x),
                conf_label,
                fontsize="small",
                horizontalalignment="center",
            )

    return axes
