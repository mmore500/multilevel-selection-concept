import functools
import itertools as it
import logging
import pprint
import sys
import typing
import warnings

from hstrat import _auxiliary_lib as hstrat_aux
from hstrat import dataframe as hstrat_df
from hstrat import hstrat
import joblib
import numpy as np
import pandas as pd
import polars as pl
from retry import retry
from scipy import stats as scipy_stats
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from tqdm import tqdm

from .._LokyBackendWithInitializer import LokyBackendWithInitializer
from .._filter_warnings import filter_warnings
from .._glimpse_df import glimpse_df
from .._mask_sequence_diffs import mask_sequence_diffs
from .._read_config import read_config
from .._screen_mutation_defined_nodes import screen_mutation_defined_nodes
from .._seed_global_rngs import seed_global_rngs
from .._shrink_df import shrink_df
from .._strong_uuid4_str import strong_uuid4_str
from .._trinomtest import trinomtest_fast


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


@_log_context_duration("_calc_tb_stats", logger=print)
def _calc_tb_stats(phylo_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:

    with hstrat_aux.log_context_duration(
        "alifestd_mask_monomorphic_clades_asexual", logger=print
    ):
        phylo_df = hstrat_aux.alifestd_mask_monomorphic_clades_asexual(
            phylo_df,
            mutate=True,
            trait_mask=phylo_df["is_leaf"].copy(),
            trait_values=phylo_df["sequence_diff"],
        )

    assert hstrat_aux.alifestd_is_working_format_asexual(phylo_df, mutate=True)
    phylo_df.reset_index(drop=True, inplace=True)

    phylo_df = hstrat_aux.alifestd_mark_sister_asexual(phylo_df, mutate=True)

    min_leaves = min(eval(cfg["cfg_clade_size_thresh"]))
    phylo_df["work_mask"] = (
        (phylo_df["num_leaves"] >= min_leaves)
        & (phylo_df["num_leaves_sibling"] >= min_leaves)
        & (
            ~phylo_df.loc[
                phylo_df["ancestor_id"].values,
                "alifestd_mask_monomorphic_clades_asexual",
            ].values
        )
    )
    phylo_df["work_mask"] |= phylo_df.loc[
        phylo_df["sister_id"].values,
        "work_mask",
    ].values  # ensure sisters of all included nodes are included

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
        "alifestd_mark_clade_fblr_growth_sister_asexual",
        logger=print,
    ):
        phylo_df = hstrat_aux.alifestd_mark_clade_fblr_growth_sister_asexual(
            phylo_df,
            mutate=True,
            parallel_backend="loky",
            progress_wrap=tqdm,
            work_mask=phylo_df["work_mask"].values.copy(),
        )
        phylo_df["clade fblr ratio"] = phylo_df["clade_fblr_growth_sister"]

    with hstrat_aux.log_context_duration(
        "alifestd_mark_clade_logistic_growth_sister_asexual",
        logger=print,
    ):
        phylo_df = (
            hstrat_aux.alifestd_mark_clade_logistic_growth_sister_asexual(
                phylo_df,
                mutate=True,
                parallel_backend=LokyBackendWithInitializer(
                    initializer=warnings.filterwarnings,
                    initargs=(
                        "ignore",  # action
                        "",  # message
                        SklearnConvergenceWarning,  # category
                    ),
                ),
                progress_wrap=tqdm,
                work_mask=phylo_df["work_mask"].values.copy(),
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


@filter_warnings(
    "ignore", category=scipy_stats._axis_nan_policy.SmallSampleWarning
)
@filter_warnings("ignore", category=RuntimeWarning)
def _calc_screen_result(
    *,
    mut_char_pos: int,
    mut_char_ref: str,
    mut_char_var: str,
    mut_freq: float,
    mut_nobs: int,
    mut_uuid: str,
    phylo_df: pd.DataFrame,
    phylo_df_background: pd.DataFrame,
    phylo_df_screened: pd.DataFrame,
    screen_min_leaves: int,
    screen_name: str,
    stat: str,
) -> typing.Dict[str, typing.Any]:

    background, screened = phylo_df_background, phylo_df_screened

    mw_U, mw_p = scipy_stats.mannwhitneyu(
        screened[stat], background[stat], alternative="two-sided"
    )
    mw_U_dropna, mw_p_dropna = scipy_stats.mannwhitneyu(
        screened[stat],
        background[stat],
        nan_policy="omit",
        alternative="two-sided",
    )
    n0, n1 = len(screened), len(background)
    cliffs_delta = 1 - 2 * (mw_U / (n0 * n1))
    cliffs_delta_dropna = 1 - 2 * (mw_U_dropna / (n0 * n1))
    binom_n = (screened[stat].dropna() != 0).sum()
    binom_k = (screened[stat].dropna() > 0).sum()
    if binom_n != 0:
        binom_result = scipy_stats.binomtest(binom_k, n=binom_n, p=0.5)
        binom_p = binom_result.pvalue
        binom_stat = binom_result.statistic
    else:
        binom_p = np.nan
        binom_stat = np.nan

    trinom_n = len(screened[stat].dropna())
    trinom_kpos = (screened[stat] > 0).sum()
    trinom_kneg = (screened[stat] < 0).sum()
    trinom_ktie = (screened[stat] == 0).sum()
    trinom_p = trinomtest_fast(screened[stat], mu=0.0, nan_policy="omit")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        trinom_stat = np.nanmean(np.sign(screened[stat]))

    screened_fill0 = screened[stat].fillna(0)
    trinom_n_fill0 = len(screened_fill0)
    trinom_kpos_fill0 = (screened_fill0 > 0).sum()
    trinom_kneg_fill0 = (screened_fill0 < 0).sum()
    trinom_ktie_fill0 = (screened_fill0 == 0).sum()
    trinom_p_fill0 = trinomtest_fast(
        screened_fill0, mu=0.0, nan_policy="raise"
    )
    trinom_stat_fill0 = np.sign(screened_fill0).mean()

    return {
        "mut": repr((mut_char_pos, mut_char_ref, mut_char_var)),
        "mut_char_pos": mut_char_pos,
        "mut_char_ref": mut_char_ref,
        "mut_char_var": mut_char_var,
        "mut_freq": mut_freq,
        "mut_nobs": mut_nobs,
        "mut_uuid": mut_uuid,
        "screen_name": screen_name,
        "screen_min_leaves": screen_min_leaves,
        "phylo_df_background_len": len(phylo_df_background),
        "phyo_df_screened_len": len(phylo_df_screened),
        "tb_stat": stat,
        "screened_nanmin": np.nanmin(
            screened[stat].values.astype(float), initial=np.inf
        ),
        "screened_nanmax": np.nanmax(
            screened[stat].values.astype(float), initial=-np.inf
        ),
        "screened_min": np.min(
            screened[stat].values.astype(float), initial=np.inf
        ),
        "screened_max": np.max(
            screened[stat].values.astype(float), initial=-np.inf
        ),
        "screened_nanmean": np.nanmean(screened[stat].values),
        "screened_nanvar": np.nanvar(screened[stat].values),
        "screened_nanstd": np.nanstd(screened[stat].values),
        "screened_nanmedian": np.nanmedian(screened[stat].values),
        "screened_mean": np.mean(screened[stat].values),
        "screened_var": np.var(screened[stat].values),
        "screened_std": np.std(screened[stat].values),
        "screened_median": np.median(screened[stat].values),
        "screened_skew": screened[stat].skew(),
        "screened_kurt": screened[stat].kurt(),
        "screened_N": len(screened),
        "screened_num_isna": screened[stat].isna().sum(),
        "screened_num_isfinite": np.isfinite(screened[stat]).sum(),
        "screened_num_notfinite": (~np.isfinite(screened[stat])).sum(),
        "screened_num_notna": screened[stat].notna().sum(),
        "screened_num_nonzero": (screened[stat] != 0).sum(),
        "screened_num_pos": (screened[stat] > 0).sum(),
        "screened_num_neg": (screened[stat] < 0).sum(),
        "screened_num_notnan": screened[stat].notna().sum(),
        "screened_num_posinf": (screened[stat] == np.inf).sum(),
        "screened_num_neginf": (screened[stat] == -np.inf).sum(),
        "screened_numnan": screened[stat].isna().sum(),
        "background_nanmin": np.nanmin(
            background[stat].values.astype(float), initial=np.inf
        ),
        "background_nanmax": np.nanmax(
            background[stat].values.astype(float), initial=-np.inf
        ),
        "background_min": np.min(
            background[stat].values.astype(float), initial=np.inf
        ),
        "background_max": np.max(
            background[stat].values.astype(float), initial=-np.inf
        ),
        "background_nanmean": np.nanmean(background[stat].values),
        "background_nanvar": np.nanvar(background[stat].values),
        "background_nanstd": np.nanstd(background[stat].values),
        "background_nanmedian": np.nanmedian(background[stat].values),
        "background_mean": np.mean(background[stat].values),
        "background_var": np.var(background[stat].values),
        "background_std": np.std(background[stat].values),
        "background_median": np.median(background[stat].values),
        "background_skew": background[stat].skew(),
        "background_kurt": background[stat].kurt(),
        "background_N": len(background),
        "background_num_isna": background[stat].isna().sum(),
        "background_num_isfinite": np.isfinite(background[stat]).sum(),
        "background_num_notfinite": (~np.isfinite(background[stat])).sum(),
        "background_num_notna": background[stat].notna().sum(),
        "background_num_nonzero": (background[stat] != 0).sum(),
        "background_num_pos": (background[stat] > 0).sum(),
        "background_num_neg": (background[stat] < 0).sum(),
        "background_num_notnan": background[stat].notna().sum(),
        "background_num_posinf": (background[stat] == np.inf).sum(),
        "background_num_neginf": (background[stat] == -np.inf).sum(),
        "background_numnan": background[stat].isna().sum(),
        "mw_U": mw_U,
        "mw_p": mw_p,
        "cliffs_delta": cliffs_delta,
        "mw_U_dropna": mw_U_dropna,
        "mw_p_dropna": mw_p_dropna,
        "cliffs_delta_dropna": cliffs_delta_dropna,
        "binom_n": binom_n,
        "binom_k": binom_k,
        "binom_p": binom_p,
        "binom_stat": binom_stat,
        "trinom_n": trinom_n,
        "trinom_kpos": trinom_kpos,
        "trinom_kneg": trinom_kneg,
        "trinom_ktie": trinom_ktie,
        "trinom_p": trinom_p,
        "trinom_stat": trinom_stat,
        "trinom_n_fill0": trinom_n_fill0,
        "trinom_kpos_fill0": trinom_kpos_fill0,
        "trinom_kneg_fill0": trinom_kneg_fill0,
        "trinom_ktie_fill0": trinom_ktie_fill0,
        "trinom_p_fill0": trinom_p_fill0,
        "trinom_stat_fill0": trinom_stat_fill0,
        **{
            c: phylo_df[c].dropna().unique().astype(str).item()
            for c in phylo_df.columns
            if (
                c.startswith("cfg_")
                or c.startswith("trt_")
                or c.startswith("replicate_")
                or c.startswith("SLURM_")
            )
        },
    }


def _process_mut(
    phylo_df: pd.DataFrame,
    cfg: dict,
    mask: np.ndarray,
    site: int,
    from_: str,
    to: str,
) -> typing.List[dict]:
    mut_uuid = strong_uuid4_str()
    # unsparsify mask
    mask_ = np.zeros(len(phylo_df), dtype=bool)
    mask_[mask] = True
    mask = mask_
    mut_nobs = mask.sum()
    mut_freq = mut_nobs / hstrat_aux.alifestd_count_leaf_nodes(phylo_df)
    assert 0 <= mut_freq <= 1

    screen_masks = screen_mutation_defined_nodes(
        phylo_df,
        has_mutation=mask,
        screens=(
            "combined_f20n50",
            "combined_f20n75",
            "naive50",
            "naive75",
            "fisher20",
            "ctrl_fisher20",
            "ctrl_naive75",
        ),
    )

    stats = (
        "clade duration ratio",
        "clade fblr ratio",
        "clade growth ratio",
        "clade size ratio",
        "num_leaves",
        "divergence_from_root",
        "origin_time",
    )
    records = []
    for stat, (screen_name, screen_mask) in it.product(
        stats, screen_masks.items()
    ):
        for screen_min_leaves in eval(cfg["cfg_clade_size_thresh"]):
            work_mask = (
                phylo_df["work_mask"].values
                & (phylo_df["num_leaves"] >= screen_min_leaves)
                & (phylo_df["num_leaves_sibling"] >= screen_min_leaves)
            )
            records.append(
                _calc_screen_result(
                    mut_char_ref=from_,
                    mut_char_pos=site,
                    mut_char_var=to,
                    mut_freq=mut_freq,
                    mut_nobs=mut_nobs,
                    mut_uuid=mut_uuid,
                    phylo_df=phylo_df,
                    phylo_df_background=phylo_df[work_mask & ~mask],
                    phylo_df_screened=phylo_df[work_mask & mask],
                    screen_min_leaves=screen_min_leaves,
                    screen_name=screen_name,
                    stat=stat,
                ),
            )

    return records


def _process_replicate(
    phylo_df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:

    phylo_df = phylo_df.copy().reset_index(drop=True)
    fil = phylo_df["sequence_diff"].str.startswith('{"0": ')
    print(
        f"{phylo_df.loc[fil, 'sequence_diff'].str.slice(0, 10).value_counts()}"
    )
    glimpse_df(phylo_df, logger=print)

    phylo_df = _prep_phylo(phylo_df, cfg)
    phylo_df = _calc_tb_stats(phylo_df, cfg)

    diffs_iter = mask_sequence_diffs(
        ancestral_sequence=phylo_df["ancestral_sequence"]
        .dropna()
        .unique()
        .astype(str)
        .item(),
        sequence_diffs=phylo_df["sequence_diff"],
        sparsify_mask=True,
        mut_count_thresh=(
            cfg["cfg_mut_count_thresh_lb"],
            cfg["cfg_mut_count_thresh_ub"],
        ),
        mut_freq_thresh=(
            cfg["cfg_mut_freq_thresh_lb"],
            cfg["cfg_mut_freq_thresh_ub"],
        ),
        mut_quant_thresh=(
            cfg["cfg_mut_quant_thresh_lb"],
            cfg["cfg_mut_quant_thresh_ub"],
        ),
        progress_wrap=tqdm,
    )

    def _process_mut_worker(*args: tuple) -> typing.List[dict]:
        return _process_mut(phylo_df, cfg, *args)

    tasks = [
        joblib.delayed(_process_mut_worker)(mask, site, frm, to)
        for (site, frm, to), mask in diffs_iter
    ]

    results = joblib.Parallel(
        n_jobs=-1,
        batch_size=10,
        backend="loky",
        verbose=50,
    )(tqdm(tasks, desc="process mutations"))

    return pd.DataFrame([*it.chain(*results)])


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
