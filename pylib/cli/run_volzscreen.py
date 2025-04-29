from contextlib import redirect_stderr, redirect_stdout
import functools
import itertools as it
import os
import pprint
import sys
import typing
import uuid
import warnings

from hstrat import _auxiliary_lib as hstrat_aux
from hstrat import dataframe as hstrat_df
from hstrat import hstrat
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats as scipy_stats
from tqdm import tqdm

from .._glimpse_df import glimpse_df
from .._mask_sequence_diffs import mask_sequence_diffs
from .._read_config import read_config
from .._screen_mutation_defined_nodes import screen_mutation_defined_nodes
from .._seed_global_rngs import seed_global_rngs
from .._shrink_df import shrink_df


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


_log_context_duration("_hsurf_fudge_phylo", logger=print)


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


_log_context_duration("_prep_phylo", logger=print)


def _prep_phylo(phylo_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:

    phylo_df["origin_time"] = phylo_df["divergence_from_root"]

    phylo_df = hstrat_aux.alifestd_to_working_format(phylo_df, mutate=False)

    assert "ancestor_id" in phylo_df.columns
    del phylo_df["ancestor_list"]

    # clean tree topology
    phylo_df = alifestd_add_inner_leaves_wf(phylo_df, mutate=True)

    phylo_df = alifestd_downsample_tips_asexual_wf(
        phylo_df, n_downsample=cfg["trt_n_downsample"]
    )

    phylo_df = alifestd_collapse_unifurcations_wf(phylo_df, mutate=True)

    # apply hstrat test drive/reconstruction
    if cfg["trt_hsurf_bits"]:
        phylo_df = _hsurf_fudge_phylo(phylo_df, cfg)

    phylo_df = alifestd_splay_polytomies_wf(phylo_df, mutate=True)
    phylo_df.drop(columns=["is_leaf"], inplace=True, errors="ignore")

    phylo_df = alifestd_collapse_unifurcations_wf(phylo_df, mutate=True)

    phylo_df = alifestd_delete_unifurcating_roots_asexual_wf(
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

    return phylo_df


_log_context_duration("_calc_tb_stats", logger=print)


def _calc_tb_stats(phylo_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    min_leaves = cfg["cfg_clade_size_thresh"]
    work_mask = (phylo_df["num_leaves"] > min_leaves) & (
        phylo_df["num_leaves_sibling"] > min_leaves
    )
    # sister statistics
    calc_dr = (
        hstrat_aux.alifestd_mark_clade_subtended_duration_ratio_sister_asexual
    )
    with hstrat_aux.log_context_duration(
        "alifestd_mark_clade_subtended_duration_ratio_sister_asexual",
        logger=print,
    ):

        phylo_df = calc_dr(phylo_df, mutate=True)
        phylo_df["clade duration ratio"] = np.log(
            phylo_df["clade_subtended_duration_ratio_sister"],
        )

    with hstrat_aux.log_context_duration(
        "alifestd_mark_clade_logistic_growth_sister_asexual",
        logger=print,
    ):
        phylo_df = (
            hstrat_aux.alifestd_mark_clade_logistic_growth_sister_asexual(
                phylo_df,
                mutate=True,
                parallel_backend="loky",
                work_mask=work_mask,
            )
        )
        phylo_df["clade growth ratio"] = phylo_df[
            "clade_logistic_growth_sister"
        ]

    with hstrat_aux.log_context_duration(
        "alifestd_mark_clade_leafcount_ratio_sister_asexual",
        logger=print,
    ):
        phylo_df = (
            hstrat_aux.alifestd_mark_clade_leafcount_ratio_sister_asexual(
                phylo_df, mutate=True
            )
        )
        phylo_df["clade size ratio"] = np.log(
            phylo_df["clade_leafcount_ratio_sister"],
        )

    return phylo_df


def _calc_screen_result(
    *,
    mut_char_pos: int,
    mut_char_ref: str,
    mut_char_var: str,
    mut_uuid: str,
    phylo_df: pd.DataFrame,
    phylo_df_background: pd.DataFrame,
    phylo_df_screened: pd.DataFrame,
    screen_name: str,
    stat: str,
) -> typing.Dict[str, typing.Any]:

    background, screened = phylo_df_background, phylo_df_screened

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=scipy_stats._axis_nan_policy.SmallSampleWarning
        )
        mw_U, mw_p = scipy_stats.mannwhitneyu(
            screened[stat], background[stat], alternative="two-sided"
        )
    n0, n1 = len(screened), len(background)
    cliffs_delta = 1 - 2 * (mw_U / (n0 * n1))
    binom_n = (screened[stat] != 0).sum()
    binom_k = (screened[stat] > 0).sum()
    if binom_n != 0:
        binom_result = scipy_stats.binomtest(binom_k, n=binom_n, p=0.5)
        binom_p = binom_result.pvalue
        binom_stat = binom_result.statistic
    else:
        binom_p = np.nan
        binom_stat = np.nan

    return {
        "mut": repr((mut_char_pos, mut_char_ref, mut_char_var)),
        "mut_char_pos": mut_char_pos,
        "mut_char_ref": mut_char_ref,
        "mut_char_var": mut_char_var,
        "mut_uuid": mut_uuid,
        "screen_name": screen_name,
        "phylo_df_background_len": len(phylo_df_background),
        "phyo_df_screened_len": len(phylo_df_screened),
        "tb_stat": stat,
        "screened_mean": screened[stat].mean(),
        "screened_var": screened[stat].var(),
        "screened_std": screened[stat].std(),
        "screened_median": screened[stat].median(),
        "screened_skew": screened[stat].skew(),
        "screened_kurt": screened[stat].kurt(),
        "screened_N": len(screened),
        "background_mean": background[stat].mean(),
        "background_var": background[stat].var(),
        "background_std": background[stat].std(),
        "background_skew": background[stat].skew(),
        "background_kurt": background[stat].kurt(),
        "background_median": background[stat].median(),
        "background_N": len(background),
        "mw_U": mw_U,
        "mw_p": mw_p,
        "cliffs_delta": cliffs_delta,
        "binom_n": binom_n,
        "binom_k": binom_k,
        "binom_p": binom_p,
        "binom_stat": binom_stat,
        **{
            c: phylo_df[c].dropna().unique().astype(str).item()
            for c in phylo_df.columns
            if (
                c.startswith("cfg_")
                or c.startswith("trt_")
                or c.startswith("replicate_")
            )
        },
    }


def _process_replicate(
    replicate_uuid: str,
    phylo_df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:

    records = []
    log_path = os.path.abspath(f"{replicate_uuid}.log")
    print(f"{log_path=}")
    with open(log_path, "w") as f, redirect_stdout(f), redirect_stderr(f):
        phylo_df = phylo_df.copy().reset_index(drop=True)
        glimpse_df(phylo_df, logger=print)

        phylo_df = _prep_phylo(phylo_df, cfg)
        phylo_df = _calc_tb_stats(phylo_df, cfg)

        min_leaves = cfg["cfg_clade_size_thresh"]
        clade_size_thresh_mask = (phylo_df["num_leaves"] > min_leaves) & (
            phylo_df["num_leaves_sibling"] > min_leaves
        )

        for (site, from_, to), mask in mask_sequence_diffs(
            ancestral_sequence=phylo_df["ancestral_sequence"]
            .dropna()
            .unique()
            .astype(str)
            .item(),
            sequence_diffs=phylo_df["sequence_diff"],
            mut_count_thresh=cfg["cfg_mut_count_thresh"],
            mut_quart_thresh=cfg["cfg_mut_quart_thresh"],
            progress_wrap=tqdm,
        ):
            mut_uuid = str(uuid.uuid4())
            screen_masks = screen_mutation_defined_nodes(
                phylo_df,
                has_mutation=mask,
            )

            stats = (
                "clade duration ratio",
                "clade growth ratio",
                "clade size ratio",
                "num_leaves",
                "divergence_from_root",
                "origin_time",
            )
            for stat, (screen_name, screen_mask) in it.product(
                stats, screen_masks.items()
            ):
                records.append(
                    _calc_screen_result(
                        mut_char_ref=from_,
                        mut_char_pos=site,
                        mut_char_var=to,
                        mut_uuid=mut_uuid,
                        phylo_df=phylo_df,
                        phylo_df_background=phylo_df[
                            clade_size_thresh_mask & ~screen_mask
                        ],
                        phylo_df_screened=phylo_df[
                            clade_size_thresh_mask & screen_mask
                        ],
                        screen_name=screen_name,
                        stat=stat,
                    ),
                )

    return pd.DataFrame(records)


if __name__ == "__main__":
    cfg = read_config(sys.stdin)
    cfg["screen_uuid"] = str(uuid.uuid4())
    pprint.PrettyPrinter(depth=4).pprint(cfg)
    seed_global_rngs(cfg["screen_num"])

    with hstrat_aux.log_context_duration("pd.read_parquet", logger=print):
        refphylos_df = pd.read_parquet(cfg["cfg_refphylos"])
        glimpse_df(refphylos_df, logger=print)

    jobs = [
        delayed(_process_replicate)(uid, phylo_df.copy(), cfg)
        for uid, phylo_df in refphylos_df.groupby(
            "replicate_uuid", observed=True
        )
    ]
    print(f"{len(jobs)=}")
    res = [*Parallel(backend="loky", n_jobs=-1, verbose=50)(jobs)]

    with hstrat_aux.log_context_duration("finalize phylo_df", logger=print):
        screen_df = pd.concat(
            res,
            ignore_index=True,
            join="outer",
        )

        for k, v in cfg.items():
            screen_df[k] = v

        screen_df = shrink_df(screen_df, inplace=True)

    glimpse_df(screen_df.head(), logger=print)
    glimpse_df(screen_df.tail(), logger=print)

    with hstrat_aux.log_context_duration("screen_df.to_parquet", logger=print):
        screen_df.to_parquet(
            f"a=run_volzscreen+screen_uuid={cfg['screen_uuid']}.pqt",
        )
