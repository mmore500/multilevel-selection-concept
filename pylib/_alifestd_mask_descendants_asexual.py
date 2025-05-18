from hstrat._auxiliary_lib import (
    alifestd_has_contiguous_ids,
    alifestd_is_topologically_sorted,
    alifestd_topological_sort,
    alifestd_try_add_ancestor_id_col,
)
import numpy as np
import pandas as pd


def alifestd_mask_descendants_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    mask: np.ndarray = None,
) -> np.ndarray:
    """Create a mask for rows containing masked nodes and their descendants.

    A topological sort will be applied if `phylogeny_df` is not topologically
    sorted. Dataframe reindexing (e.g., df.index) may be applied.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.
    """

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    if not alifestd_is_topologically_sorted(phylogeny_df):
        phylogeny_df = alifestd_topological_sort(phylogeny_df, mutate=True)

    if alifestd_has_contiguous_ids(phylogeny_df):
        phylogeny_df.reset_index(drop=True, inplace=True)
    else:
        phylogeny_df.index = phylogeny_df["id"]

    phylogeny_df["alifestd_mask_descendants_asexual"] = mask

    for idx in phylogeny_df.index:
        ancestor_id = phylogeny_df.at[idx, "ancestor_id"]

        phylogeny_df.at[idx, "alifestd_mask_descendants_asexual"] = (
            phylogeny_df.at[ancestor_id, "alifestd_mask_descendants_asexual"]
            | phylogeny_df.at[idx, "alifestd_mask_descendants_asexual"]
        )

    return phylogeny_df["alifestd_mask_descendants_asexual"].to_numpy()
