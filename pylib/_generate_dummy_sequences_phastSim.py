import random
import typing

from hstrat import _auxiliary_lib as hstrat_aux
import pandas as pd

from ._run_phastSim import run_phastSim
from ._seed_global_rngs import seed_global_rngs


def _worker(
    args: typing.Tuple[int, pd.DataFrame, typing.Dict[str, str]],
) -> pd.DataFrame:
    """Worker for phastSim simulation."""
    flavor_origin, group_df, ancestral_sequences, seed = args
    seed_global_rngs(seed)
    group_df = group_df.copy().reset_index(drop=True)
    group_df.loc[
        group_df["id"] == flavor_origin, "ancestor_id"
    ] = flavor_origin
    variant_flavor = group_df["variant_flavor"].unique().item()
    ancestral_sequence = ancestral_sequences[variant_flavor]

    group_df = hstrat_aux.alifestd_collapse_unifurcations(
        group_df, mutate=True
    )
    group_df["taxon_label"] = group_df["id"]
    group_df = hstrat_aux.alifestd_to_working_format(group_df, mutate=True)

    return run_phastSim(
        ancestral_sequence=ancestral_sequence,
        phylogeny_df=group_df,
        taxon_label="taxon_label",
    )


def generate_dummy_sequences_phastSim(
    phylogeny_df: pd.DataFrame,
    ancestral_sequences: typing.Dict[str, str],  # variant flavor -> sequence
    progress_map: typing.Callable = map,
) -> pd.DataFrame:
    """Generate dummy sequences based on a phylogeny DataFrame and an ancestral
    sequence.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        Alife standard DataFrame containing the phylogenetic relationships.

        Must be topologically sorted.

    ancestral_sequence : str
        The ancestral sequence.

    progress_map : typing.Callable, default map
        Pass tqdm.contrib.concurrent.process_map or equivalent to display
        progress bar and use multiprocessing.

    Returns
    -------
    pd.DataFrame
        Dummy sequences generated based on the phylogeny and ancestral
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

    phylogeny_df["is_variant_flavor_origin"] = (
        phylogeny_df["variant_flavor"].values
        != phylogeny_df.loc[
            phylogeny_df["ancestor_id"], "variant_flavor"
        ].values
    ) | (phylogeny_df["ancestor_id"].values == phylogeny_df["id"].values)

    origins = []
    for idx in phylogeny_df.index:
        ancestor_id = phylogeny_df.at[idx, "ancestor_id"]
        is_variant_flavor_origin = phylogeny_df.at[
            idx, "is_variant_flavor_origin"
        ]
        origins.append(
            idx if is_variant_flavor_origin else origins[ancestor_id]
        )

    phylogeny_df["variant_flavor_origin"] = origins

    groups = list(phylogeny_df.groupby("variant_flavor_origin", sort=False))
    args = [
        (
            flavor_origin,
            group_df,
            ancestral_sequences,
            random.getrandbits(32),
        )
        for flavor_origin, group_df in groups
    ]

    generated_sequences = progress_map(_worker, args)

    return pd.concat(generated_sequences, ignore_index=True)
