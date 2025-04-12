from hstrat import _auxiliary_lib as hstrat_aux
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

    screen_trait_defined_naive = (
        hstrat_aux.alifestd_screen_trait_defined_clades_naive_asexual(
            phylo_df,
            mutate=True,
            mask_trait_absent=(~has_mutation) & phylo_df["is_leaf"],
            mask_trait_present=has_mutation & phylo_df["is_leaf"],
        )
    )
    return {
        "combined": screen_trait_defined_fisher & screen_trait_defined_naive,
        "fisher": screen_trait_defined_fisher,
        "naive": screen_trait_defined_naive,
    }
