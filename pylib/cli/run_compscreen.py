import functools
import logging
import pprint
import sys
import typing

from hstrat import _auxiliary_lib as hstrat_aux
from hstrat import dataframe as hstrat_df
from hstrat import hstrat
import pandas as pd
import polars as pl
from retry import retry
from tqdm import tqdm

from .._calc_normed_defmut_clade_stats import calc_normed_defmut_clade_stats
from .._glimpse_df import glimpse_df
from .._mask_sequence_diffs import mask_sequence_diffs
from .._read_config import read_config
from .._screen_mutation_defined_nodes import screen_mutation_defined_nodes
from .._screen_mutation_defined_nodes_sequence_diff import (
    screen_mutation_defined_nodes_sequence_diff,
)
from .._seed_global_rngs import seed_global_rngs
from .._shrink_df import shrink_df
from .._strong_uuid4_str import strong_uuid4_str


# have to redefine for joblib compat
def _log_context_duration(what: str, logger: typing.Callable = print):
    def decorator(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with hstrat_aux.log_context_duration(what, logger=logger):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def _wtwf(wrapee: typing.Callable) -> typing.Callable:
    @functools.wraps(wrapee)
    def decorated(phylo_df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        with hstrat_aux.log_context_duration(wrapee.__name__, logger=print):
            # convert to working format
            phylo_df = hstrat_aux.alifestd_to_working_format(
                phylo_df, mutate=False
            )
            return hstrat_aux.alifestd_to_working_format(
                wrapee(phylo_df, *args, **kwargs),
                mutate=True,
            )

    return decorated


alifestd_add_inner_leaves_wf = _wtwf(hstrat_aux.alifestd_add_inner_leaves)
alifestd_collapse_unifurcations_wf = _wtwf(
    hstrat_aux.alifestd_collapse_unifurcations,
)
alifestd_delete_unifurcating_roots_asexual_wf = _wtwf(
    hstrat_aux.alifestd_delete_unifurcating_roots_asexual,
)
alifestd_downsample_tips_asexual_wf = _wtwf(
    hstrat_aux.alifestd_downsample_tips_asexual,
)
alifestd_join_roots_wf = _wtwf(hstrat_aux.alifestd_join_roots)
alifestd_splay_polytomies_wf = _wtwf(hstrat_aux.alifestd_splay_polytomies)


@_log_context_duration("_hsurf_fudge_phylo", logger=print)
def _hsurf_fudge_phylo(phylo_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:

    with hstrat_aux.log_context_duration(
        "hstrat_df.surface_test_drive", logger=print
    ):
        pop_df = hstrat_df.surface_test_drive(
            pl.from_pandas(phylo_df),
            dstream_algo="dstream.primed_0pad0_tiltedxtc_algo",
            dstream_S=cfg["trt_hsurf_bits"],
            stratum_differentia_bit_width=1,
            progress_wrap=tqdm,
        )

    with hstrat_aux.log_context_duration(
        "hstrat_df.surface_build_tree", logger=print
    ):
        phylo_df = hstrat_df.surface_build_tree(
            pop_df,
            delete_trunk=True,
            trie_postprocessor=hstrat.AssignOriginTimeNodeRankTriePostprocessor(
                t0="dstream_S",
            ),
        ).to_pandas()

    return alifestd_join_roots_wf(phylo_df, mutate=True)


@_log_context_duration("_prep_phylo", logger=print)
def _prep_phylo(phylo_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:

    phylo_df.drop(
        columns=["is_leaf", "is_root", "node_depth", "num_children"],
        errors="ignore",
        inplace=True,
    )

    phylo_df["origin_time"] = phylo_df["divergence_from_root"]

    phylo_df = hstrat_aux.alifestd_to_working_format(phylo_df, mutate=False)

    assert "ancestor_id" in phylo_df.columns
    del phylo_df["ancestor_list"]

    # clean tree topology
    phylo_df = alifestd_downsample_tips_asexual_wf(
        phylo_df, n_downsample=cfg["trt_n_downsample"]
    )

    phylo_df = alifestd_collapse_unifurcations_wf(phylo_df, mutate=True)

    # apply hstrat test drive/reconstruction
    if cfg["trt_hsurf_bits"]:
        phylo_df = _hsurf_fudge_phylo(phylo_df, cfg)

    phylo_df = alifestd_collapse_unifurcations_wf(phylo_df, mutate=True)
    phylo_df = alifestd_delete_unifurcating_roots_asexual_wf(
        phylo_df, mutate=True
    )
    phylo_df = alifestd_splay_polytomies_wf(phylo_df, mutate=True)
    assert hstrat_aux.alifestd_is_strictly_bifurcating_asexual(
        phylo_df, mutate=True
    )

    # more statistics
    phylo_df = hstrat_aux.alifestd_mark_leaves(phylo_df, mutate=True)
    phylo_df = hstrat_aux.alifestd_mark_num_leaves_asexual(
        phylo_df, mutate=True
    )
    phylo_df = hstrat_aux.alifestd_mark_num_leaves_sibling_asexual(
        phylo_df, mutate=True
    )
    phylo_df = hstrat_aux.alifestd_mark_roots(phylo_df, mutate=True)

    phylo_df.drop(
        columns=["is_leaf", "is_root", "node_depth", "num_children"],
        errors="ignore",
        inplace=True,
    )

    return phylo_df


def _process_replicate(
    phylo_df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:

    try:
        ancestral_sequence = (
            phylo_df["ancestral_sequence"].astype(str).unique().item()
        )
    except ValueError:
        print(phylo_df["ancestral_sequence"].value_counts())
        raise

    phylo_df = phylo_df.copy().reset_index(drop=True)

    phylo_df = _prep_phylo(phylo_df, cfg)

    # yield (mut_char_pos, mut_char_ref, mut_char_var), mut_mask
    mutations = mask_sequence_diffs(
        ancestral_sequence=ancestral_sequence,
        sequence_diffs=phylo_df["sequence_diff"],
        sparsify_mask=False,
    )
    mutations = [*mutations]
    phylo_df["has_focal_mutation"] = mutations[0][1]

    if cfg["trt_hsurf_bits"] == 0:
        phylo_df["screen_name"] = "sequence_diff"
        defining_masks = {
            (
                mut_char_pos,
                mut_char_ref,
                mut_char_var,
            ): screen_mutation_defined_nodes_sequence_diff(
                phylo_df=phylo_df,
                mut_char_pos=mut_char_pos,
                mut_char_var=mut_char_var,
            )
            for (
                mut_char_pos,
                mut_char_ref,
                mut_char_var,
            ), mut_mask in mutations
        }
    else:
        phylo_df["screen_name"] = "naive50"
        defining_masks = {
            (
                mut_char_pos,
                mut_char_ref,
                mut_char_var,
            ): screen_mutation_defined_nodes(
                phylo_df,
                has_mutation=mut_mask,
                screens=["naive50"],
            )[
                "naive50"
            ]
            for (
                mut_char_pos,
                mut_char_ref,
                mut_char_var,
            ), mut_mask in mutations
        }

    phylo_df["is_focal_defmut"] = next(iter(defining_masks.values()))

    return calc_normed_defmut_clade_stats(
        phylo_df=phylo_df,
        defmut_clade_masks=defining_masks,
        match_cols=["variant_flavor"],
        ot_deltas=(4, 7, 14, 28, 44),
        progress_wrap=tqdm,
    )


def main(refphylos_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cfg = cfg.copy()

    pprint.PrettyPrinter(depth=4).pprint(cfg)
    seed_global_rngs(cfg["screen_num"])

    work = [
        phylo_df
        for uid, phylo_df in refphylos_df.groupby(
            "replicate_uuid", observed=True
        )
        if "cfg_assigned_replicate_uuid" not in cfg
        or str(uid) == cfg["cfg_assigned_replicate_uuid"]
    ]
    res = [
        _process_replicate(phylo_df, cfg)
        for phylo_df in tqdm(work, desc="process replicate")
    ]

    with hstrat_aux.log_context_duration("finalize phylo_df", logger=print):
        screen_df = pd.concat(res)

        for k, v in cfg.items():
            screen_df[k] = v

        screen_df = shrink_df(screen_df, inplace=True)

    return screen_df


if __name__ == "__main__":
    hstrat_aux.configure_prod_logging()
    cfg = read_config(sys.stdin)
    cfg["screen_uuid"] = strong_uuid4_str()

    with hstrat_aux.log_context_duration("pd.read_parquet", logger=print):
        read_parquet = retry(tries=5, logger=logging.getLogger(__name__))(
            pd.read_parquet
        )
        refphylos_df = read_parquet(cfg["cfg_refphylos"])
        glimpse_df(refphylos_df, logger=print)

    screen_df = main(refphylos_df, cfg)

    glimpse_df(screen_df.head(), logger=print)
    glimpse_df(screen_df.tail(), logger=print)

    with hstrat_aux.log_context_duration("screen_df.to_parquet", logger=print):
        screen_df.to_parquet(
            f"a=run_volzscreen+screen_uuid={cfg['screen_uuid']}.pqt",
        )
