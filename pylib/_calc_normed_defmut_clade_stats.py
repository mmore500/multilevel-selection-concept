import typing

from hstrat import _auxiliary_lib as hstrat_aux
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def calc_normed_defmut_clade_stats(
    phylo_df: pd.DataFrame,
    defmut_clade_masks: dict[tuple[int, str, str], np.ndarray],
    stat_cols: tuple[str] = ("num_leaves", "clade_duration"),
    match_cols: tuple[str] = tuple(),
    ot_deltas: tuple[float] = tuple(),
    progress_wrap: typing.Callable = lambda x: x,
) -> pd.DataFrame:

    phylo_df = hstrat_aux.alifestd_to_working_format(phylo_df, mutate=True)

    if "num_leaves" in stat_cols:
        phylo_df = hstrat_aux.alifestd_mark_num_leaves_asexual(
            phylo_df, mutate=True
        )

    if "clade_duration" in stat_cols:
        phylo_df = hstrat_aux.alifestd_mark_clade_duration_asexual(
            phylo_df, mutate=True
        )

    assert all(len(v) == len(phylo_df) for v in defmut_clade_masks.values())
    mask_sum = sum(defmut_clade_masks.values())
    phylo_df["defmut_mask_sum"] = mask_sum

    subnorm_masks = {
        "all": lambda row: np.ones_like(mask_sum, dtype=bool),
        **{  # force early binding of ot_delta
            f"ot_delta:{ot_delta}": lambda row, ot_delta=ot_delta: (
                np.abs(
                    phylo_df["origin_time"].values[row]
                    - phylo_df["origin_time"].values
                )
                < ot_delta
            )
            for ot_delta in ot_deltas
        },
        **{  # force early binding of match_col
            f"match:{match_col}": lambda row, match_col=match_col: (
                phylo_df[match_col].values[row] == phylo_df[match_col].values
            )
            for match_col in match_cols
        },
    }

    for subnorm_name, subnorm_mask in progress_wrap(subnorm_masks.items()):
        num_comparators = -np.ones_like(mask_sum, dtype=np.int64)
        norm_result = {
            key: np.full_like(mask_sum, np.nan, dtype=np.float64)
            for key in stat_cols
        }
        comparator_median = {
            key: np.full_like(mask_sum, np.nan, dtype=np.float64)
            for key in stat_cols
        }
        comparator_mean = {
            key: np.full_like(mask_sum, np.nan, dtype=np.float64)
            for key in stat_cols
        }
        comparator_std = {
            key: np.full_like(mask_sum, np.nan, dtype=np.float64)
            for key in stat_cols
        }
        for row in progress_wrap(np.flatnonzero(mask_sum)):
            assert mask_sum[row]
            comparator_mask = (
                mask_sum.astype(bool)
                & subnorm_mask(row).astype(bool)
                & (phylo_df["id"].values != row)
            )
            num_comparators[row] = comparator_mask.sum()

            for stat_col in stat_cols:
                comparator_values = phylo_df.loc[
                    comparator_mask.astype(bool), stat_col
                ].values
                assert len(comparator_values) == num_comparators[row]
                norm_result[stat_col][row] = scipy_stats.percentileofscore(
                    comparator_values,
                    phylo_df[stat_col].values[row],
                    kind="mean",
                )
                comparator_median[stat_col][row] = np.median(comparator_values)
                comparator_mean[stat_col][row] = np.mean(comparator_values)
                comparator_std[stat_col][row] = np.std(comparator_values)

        for stat_col, v in norm_result.items():
            phylo_df[f"defmut_norm_{subnorm_name}-{stat_col}"] = v
        for stat_col, v in comparator_median.items():
            phylo_df[f"defmut_norm_{subnorm_name}-{stat_col}_median"] = v
        for stat_col, v in comparator_mean.items():
            phylo_df[f"defmut_norm_{subnorm_name}-{stat_col}_mean"] = v
        for stat_col, v in comparator_std.items():
            phylo_df[f"defmut_norm_{subnorm_name}-{stat_col}_std"] = v

        phylo_df[
            f"defmut_norm_{subnorm_name}-num_comparators"
        ] = num_comparators

    return phylo_df
