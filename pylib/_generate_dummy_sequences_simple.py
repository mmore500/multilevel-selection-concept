import itertools as it
import random
import typing

from hstrat import _auxiliary_lib as hstrat_aux
import numba as nb
import numpy as np
import pandas as pd
import polars as pl

from ._seed_global_rngs import seed_global_rngs


@nb.njit(cache=True)
def _do_sequences(
    ancestral_arr: np.ndarray,
    ancestor_ids: np.ndarray,
    origin_time_deltas: np.ndarray,
) -> np.ndarray:
    n = len(ancestral_arr)
    N = len(ancestor_ids)
    shape = (N, n)
    arrs = np.zeros(shape, "uint8")

    for idx in range(len(ancestor_ids)):
        ancestor_id = ancestor_ids[idx]
        if ancestor_id == idx:
            arrs[idx, :] = ancestral_arr
            continue

        ot_delta = origin_time_deltas[idx]

        rand1to3 = (np.random.rand(n) * 3.0).astype(np.uint8) + 1
        muts = rand1to3 * (
            np.random.rand(n) < ot_delta * 2.74e-6  # this is approximation
        )
        arrs[idx, :] = (arrs[ancestor_id, :] + muts) & 3

    return arrs


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

    # temporarily remove "-" characters
    with hstrat_aux.log_context_duration("remove dashes", print):
        dash_indices = [
            i for i, char in enumerate(ancestral_sequence) if char == "-"
        ]
        ancestral_sequence_ = ancestral_sequence
        ancestral_sequence = ancestral_sequence.replace("-", "")

    # check for whitespace
    if ancestral_sequence != "".join(ancestral_sequence.split()):
        raise ValueError("Ancestral sequence contains whitespace")

    group_df = hstrat_aux.alifestd_collapse_unifurcations(
        group_df, mutate=True
    )
    group_df["taxon_label"] = group_df["id"]
    group_df = hstrat_aux.alifestd_to_working_format(group_df, mutate=True)
    group_df = hstrat_aux.alifestd_mark_origin_time_delta_asexual(
        group_df, mutate=True
    )
    group_df = hstrat_aux.alifestd_mark_leaves(group_df, mutate=True)

    ancestral_arr = np.array(
        [
            *map(
                {"A": 0, "C": 1, "G": 2, "T": 3}.__getitem__,
                ancestral_sequence,
            ),
        ],
        dtype=np.uint8,
    )

    with hstrat_aux.log_context_duration("_do_sequences", print):
        arrs = _do_sequences(
            ancestral_arr,
            group_df["ancestor_id"].to_numpy(dtype=np.uint32),
            group_df["origin_time"].to_numpy(dtype=np.float64),
        )

    with hstrat_aux.log_context_duration("extract", print):
        int_to_base = np.array(["A", "C", "G", "T"], dtype="<U1")

        leaf_arrs = arrs[group_df["is_leaf"].values, :]

        # map ints -> bases to get a 2D array of single-character strings
        # shape is (n_leaves, L), dtype='<U1'
        char_arr = int_to_base[leaf_arrs]

        # now reinterpret each row of L characters as a Unicode string
        L = char_arr.shape[1]
        leaf_strs = char_arr.view(f"U{L}")[:, 0]

    res = pl.DataFrame(
        {
            "id": group_df.loc[
                group_df["is_leaf"].values, "taxon_label"
            ].values,
            "sequence": leaf_strs,
        },
    )

    with hstrat_aux.log_context_duration("restore dashes", print):
        segments = [
            pl.col("sequence").str.slice(
                # shift start forward by 1 for every dash removed (i > 0)
                apos - i + int(bool(i)),
                # length is the gap between dashes
                bpos - apos - int(bool(i)),
            )
            for i, (apos, bpos) in enumerate(
                it.pairwise(
                    [
                        0,
                        *dash_indices,
                        len(ancestral_sequence_),
                    ],
                )
            )
        ]
        res = (
            res.lazy()
            .with_columns(sequence=pl.concat_str(segments, separator="-"))
            .collect()
        )

    assert res["sequence"].str.len_chars().unique().item() == len(
        ancestral_sequence_,
    )

    assert len(res) == hstrat_aux.alifestd_count_leaf_nodes(group_df)
    assert all(res["sequence"].first()[pos] == "-" for pos in dash_indices)

    return res.to_pandas()


def generate_dummy_sequences_simple(
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
