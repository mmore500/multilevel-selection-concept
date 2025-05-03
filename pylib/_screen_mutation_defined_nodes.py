from hstrat import _auxiliary_lib as hstrat_aux
import numpy as np
import pandas as pd


def screen_mutation_defined_nodes(
    phylo_df: pd.DataFrame,
    has_mutation: pd.Series,
) -> dict:
    phylo_df.reset_index(drop=True, inplace=True)

    # trait screening --- trait-defined
    screen_trait_defined_fisher = (
        hstrat_aux.alifestd_screen_trait_defined_clades_fisher_asexual(
            phylo_df,
            mutate=True,
            mask_trait_absent=(~has_mutation) & phylo_df["is_leaf"],
            mask_trait_present=has_mutation & phylo_df["is_leaf"],
        )
        < 0.2
    )
    # pick furthest-down significant node, for consecutive clades
    # i..e, if there is a pocket of trait-having leaves, several inner nodes
    # above may test as significant, but we want the furthest down
    assert hstrat_aux.alifestd_is_working_format_asexual(phylo_df)
    screen_trait_defined_fisher[
        phylo_df["ancestor_id"].to_numpy()
    ] &= ~screen_trait_defined_fisher

    screen_trait_defined_naive50 = (
        hstrat_aux.alifestd_screen_trait_defined_clades_naive_asexual(
            phylo_df,
            mutate=True,
            mask_trait_absent=(~has_mutation) & phylo_df["is_leaf"],
            mask_trait_present=has_mutation & phylo_df["is_leaf"],
            defining_mut_thresh=0.50,
            defining_mut_sister_thresh=0.50,
        )
    )
    screen_trait_defined_naive75 = (
        hstrat_aux.alifestd_screen_trait_defined_clades_naive_asexual(
            phylo_df,
            mutate=True,
            mask_trait_absent=(~has_mutation) & phylo_df["is_leaf"],
            mask_trait_present=has_mutation & phylo_df["is_leaf"],
            defining_mut_thresh=0.75,
            defining_mut_sister_thresh=0.75,
        )
    )

    fisher = screen_trait_defined_fisher
    naive50 = screen_trait_defined_naive50
    naive75 = screen_trait_defined_naive75
    combined50 = fisher & naive50
    combined75 = fisher & naive75
    return {
        "combined50": combined50,
        "combined75": combined75,
        "naive50": naive50,
        "naive75": naive75,
        "fisher": fisher,
        "ctrl_fisher": np.random.permutation(fisher.copy()),
        "ctrl_naive75": np.random.permutation(naive75.copy()),
    }
