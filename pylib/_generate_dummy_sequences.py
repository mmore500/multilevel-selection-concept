import typing

from hstrat import _auxiliary_lib as hstrat_aux
import pandas as pd


def generate_dummy_sequences(
    phylogeny_df: pd.DataFrame,
    ancestral_sequence: str,
    mutator: typing.Callable,
) -> typing.List[str]:
    """Generate dummy sequences based on a phylogeny DataFrame and an ancestral
    sequence.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        Alife standard DataFrame containing the phylogenetic relationships.

        Must be topologically sorted.

    ancestral_sequence : str
        The ancestral sequence.

    Returns
    -------
    list of str
        List of dummy sequences generated based on the phylogeny and ancestral
        sequence.
    """

    if not hstrat_aux.alifestd_is_topologically_sorted(phylogeny_df):
        raise ValueError("Phylogeny DataFrame is not topologically sorted.")

    if not hstrat_aux.alifestd_has_contiguous_ids(phylogeny_df):
        raise ValueError("Phylogeny DataFrame does not have contiguous IDs.")

    if "origin_time" not in phylogeny_df.columns:
        raise ValueError(
            "Phylogeny DataFrame does not have an origin_time column.",
        )

    phylogeny_df = hstrat_aux.alifestd_try_add_ancestor_id_col(
        phylogeny_df, mutate=True
    )

    phylogeny_df.reset_index(drop=True, inplace=True)

    res = []
    for idx in phylogeny_df.index:
        ancestor_id = phylogeny_df.at[idx, "ancestor_id"]

        if ancestor_id == idx:
            res.append(ancestral_sequence)
            continue

        ancestor_sequence = res[ancestor_id]
        mutated_sequence = mutator(
            ancestor_sequence,
            variant=phylogeny_df.at[idx, "variant"],
            ancestor_variant=phylogeny_df.at[ancestor_id, "variant"],
            origin_time=phylogeny_df.at[idx, "origin_time"],
            ancestor_origin_time=phylogeny_df.at[ancestor_id, "origin_time"],
        )

        res.append(mutated_sequence)

    return res
