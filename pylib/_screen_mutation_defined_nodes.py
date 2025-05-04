import typing

from hstrat import _auxiliary_lib as hstrat_aux
import numpy as np
import pandas as pd


def screen_mutation_defined_nodes(
    phylo_df: pd.DataFrame,
    has_mutation: pd.Series,
    screens: typing.Sequence[str] = ("combined", "naive", "fisher"),
) -> dict:
    phylo_df.reset_index(drop=True, inplace=True)

    trait_absent = (~has_mutation) & phylo_df["is_leaf"].values
    trait_present = has_mutation & phylo_df["is_leaf"].values

    # trait screening --- trait-defined
    fisher20 = (
        hstrat_aux.alifestd_screen_trait_defined_clades_fisher_asexual(
            phylo_df,
            mutate=True,
            mask_trait_absent=trait_absent.copy(),
            mask_trait_present=trait_present.copy(),
        )
        < 0.2
    ).copy()
    # pick furthest-down significant node, for consecutive clades
    # i..e, if there is a pocket of trait-having leaves, several inner nodes
    # above may test as significant, but we want the furthest down
    assert hstrat_aux.alifestd_is_working_format_asexual(phylo_df)
    fisher20[phylo_df["ancestor_id"].values] &= ~fisher20

    naive50 = (
        hstrat_aux.alifestd_screen_trait_defined_clades_naive_asexual(
            phylo_df,
            mutate=True,
            mask_trait_absent=trait_absent.copy(),
            mask_trait_present=trait_present.copy(),
            defining_mut_thresh=0.50,
            defining_mut_sister_thresh=0.50,
        )
    ).copy()
    naive75 = (
        hstrat_aux.alifestd_screen_trait_defined_clades_naive_asexual(
            phylo_df,
            mutate=True,
            mask_trait_absent=trait_absent.copy(),
            mask_trait_present=trait_present.copy(),
            defining_mut_thresh=0.75,
            defining_mut_sister_thresh=0.75,
        )
    ).copy()

    recipes = {
        "combined": lambda: fisher20 & naive75,
        "combined_f20n50": lambda: fisher20 & naive50,
        "combined_f20n75": lambda: fisher20 & naive75,
        "naive": lambda: naive75,
        "naive50": lambda: naive50,
        "naive75": lambda: naive75,
        "fisher": lambda: fisher20,
        "fisher20": lambda: fisher20,
        "ctrl_fisher20": lambda: np.random.permutation(fisher20.copy()),
        "ctrl_naive75": lambda: np.random.permutation(naive75.copy()),
    }

    return {item: recipes[item]().copy() for item in screens}
