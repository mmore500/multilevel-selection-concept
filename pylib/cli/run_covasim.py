from collections import defaultdict
import logging
import pprint
import random
import sys
import typing

import covasim as cv
from hstrat import _auxiliary_lib as hstrat_aux
import numpy as np
import pandas as pd
from retry import retry
from tqdm import tqdm

from .._SyncHostCompartments import SyncHostCompartments
from .._SyncHostCompartmentsBackground import SyncHostCompartmentsBackground
from .._VariantFlavor import VariantFlavor
from .._cv_infection_log_to_alstd_df import cv_infection_log_to_alstd_df
from .._diff_sequences import diff_sequences
from .._glimpse_df import glimpse_df
from .._make_cv_sim_uk import make_cv_sim_uk
from .._make_cv_sim_vanilla import make_cv_sim_vanilla
from .._make_flavored_variants import make_flavored_variants
from .._make_variant_flavors import make_variant_flavors
from .._make_wt_specs_single import make_wt_specs_single
from .._make_wt_specs_uk import make_wt_specs_uk
from .._read_config import read_config
from .._seed_global_rngs import seed_global_rngs
from .._shrink_df import shrink_df
from .._strong_uuid4_str import strong_uuid4_str


@retry(tries=5, logger=logging.getLogger(__name__))
def _get_reference_sequences(
    cfg: dict,
) -> typing.Dict[str, str]:
    reference_sequences = pd.read_csv(cfg["cfg_refseqs"])
    return dict(
        zip(
            reference_sequences["WHO Label"].values,
            # remove whitespace pollution
            # and only use first N characters of the sequence, for perf/memory
            reference_sequences["Aligned Sequence"]
            .str.replace(r"\s+", "", regex=True)
            .str.slice(0, cfg.get("cfg_maxseqlen", None))
            .str.slice(0, cfg.get("trt_maxseqlen", None))
            .values,
        ),
    )


def _setup_sim(
    cfg: dict,
    *,
    reference_sequences: typing.Dict[str, str],
) -> typing.Tuple[cv.Sim, typing.List[VariantFlavor]]:
    mutmx_variant = defaultdict(lambda: 1)
    mutmx_variant["rel_beta"] = cfg["trt_mutmx_rel_beta"]
    # rel_symp_prob
    # rel_severe_prob
    # rel_crit_prob
    # rel_death_prob

    make_wt_specs = {
        "make_wt_specs_uk": make_wt_specs_uk,
        "make_wt_specs_single": make_wt_specs_single,
    }[cfg["cfg_make_wt_specs_recipe"]]

    wt_specs = make_wt_specs(
        reference_sequences=reference_sequences,
    )
    variant_flavors = make_variant_flavors(
        wt_specs,
        mut_variant=lambda x: {
            k: v * mutmx_variant[k] for k, v in x.variant.items()
        },
        mut_withinhost_r=lambda x: (
            x.withinhost_r * cfg["trt_mutmx_withinhost_r"]
        ),
        mut_active_strain_factor=lambda x: (
            x.active_strain_factor * cfg["trt_mutmx_active_strain_factor"]
        ),
        p_wt_to_mut=lambda __: cfg["cfg_p_wt_to_mut"]
        * cfg["cfg_num_mut_sites"],
        suffix_mut=cfg["cfg_suffix_mut"],
        suffix_wt=cfg["cfg_suffix_wt"],
    )
    flavored_variants = make_flavored_variants(variant_flavors)

    make_sim = {
        "make_cv_sim_uk": make_cv_sim_uk,
        "make_cv_sim_vanilla": make_cv_sim_vanilla,
    }[cfg["cfg_make_cv_sim_recipe"]]

    return (
        make_sim(
            preinterventions=[
                SyncHostCompartments(
                    variant_flavors=variant_flavors,
                    pop_size=cfg["cfg_pop_size"],
                ),
                SyncHostCompartmentsBackground(
                    variant_flavors=variant_flavors,
                    pop_size=cfg["cfg_pop_size"],
                    num_background_strains=cfg["cfg_maxseqlen"],
                ),
            ],
            variants=flavored_variants,
            pop_size=cfg["cfg_pop_size"],
            seed=cfg["trt_seed"],
        ),
        variant_flavors,
    )


def _extract_phylo(
    infection_log: dict,
    variant_flavors: typing.List[VariantFlavor],
) -> pd.DataFrame:
    with hstrat_aux.log_context_duration(
        "cv_infection_log_to_alstd_df", logger=print
    ):
        phylo_df = cv_infection_log_to_alstd_df(
            infection_log, join_roots=False
        )

    with hstrat_aux.log_context_duration("map variant_flavor", logger=print):
        phylo_df["variant_flavor"] = phylo_df["variant"].map(
            {
                v.label: vf.label
                for vf in variant_flavors
                for v in (vf.variant_mut, vf.variant_wt)
            },
        )

    phylo_df = hstrat_aux.alifestd_to_working_format(phylo_df, mutate=True)
    phylo_df.reset_index(drop=True, inplace=True)
    assert (
        phylo_df.loc[phylo_df["ancestor_id"], "variant_flavor"].values
        == phylo_df["variant_flavor"].values
    ).all()

    with hstrat_aux.log_context_duration("alifestd_join_roots", logger=print):
        variant_dfs = [
            hstrat_aux.alifestd_join_roots(group_df, mutate=False)
            for __, group_df in phylo_df.groupby(
                "variant_flavor",
                as_index=False,
                observed=True,
            )
        ]
        assert all(
            hstrat_aux.alifestd_validate(
                hstrat_aux.alifestd_try_add_ancestor_list_col(df),
            )
            for df in variant_dfs
        )

        phylo_df_ = hstrat_aux.alifestd_join_roots(
            pd.concat(variant_dfs, ignore_index=True), mutate=True
        )
        assert len(phylo_df) == len(phylo_df_)
        phylo_df = phylo_df_
        assert hstrat_aux.alifestd_validate(
            hstrat_aux.alifestd_try_add_ancestor_list_col(phylo_df),
        )

    with hstrat_aux.log_context_duration("alifestd_add_inner_leaves_asexual"):
        phylo_df = hstrat_aux.alifestd_add_inner_leaves_asexual(
            phylo_df, mutate=True
        )

    with hstrat_aux.log_context_duration(
        "alifestd_to_working_format", logger=print
    ):
        phylo_df = hstrat_aux.alifestd_to_working_format(phylo_df, mutate=True)

    return phylo_df


def _add_sequence_diffs(phylo_df: pd.DataFrame) -> pd.DataFrame:
    assert hstrat_aux.alifestd_is_topologically_sorted(phylo_df)
    assert hstrat_aux.alifestd_count_root_nodes(phylo_df) == 1

    phylo_df = hstrat_aux.alifestd_mark_node_depth_asexual(
        phylo_df, mutate=True
    )
    phylo_df = hstrat_aux.alifestd_mark_leaves(phylo_df, mutate=True)

    ancestral_sequence = (
        phylo_df.loc[phylo_df["is_leaf"]]
        .sort_values(by="node_depth")["sequence"]
        .iat[0]
    )
    assert len(ancestral_sequence) == phylo_df["sequence"].str.len().max()

    phylo_df["ancestral_sequence"] = ancestral_sequence
    assert phylo_df["sequence"].dropna().str.len().nunique() == 1

    phylo_df["sequence_diff"] = diff_sequences(
        phylo_df["sequence"],
        ancestral_sequence=ancestral_sequence,
        progress_wrap=tqdm,
    )
    del phylo_df["sequence"]

    assert phylo_df["sequence_diff"].str.len().fillna(0).all()

    return phylo_df


def main(cfg: dict) -> pd.DataFrame:
    cfg = cfg.copy()

    pprint.PrettyPrinter(depth=4).pprint(cfg)
    seed_global_rngs(cfg["trt_seed"])
    cfg["py_random_sample1"] = random.getrandbits(32)
    cfg["np_random_sample1"] = np.random.randint(2**32)

    reference_sequences = _get_reference_sequences(cfg)
    sim, variant_flavors = _setup_sim(
        cfg, reference_sequences=reference_sequences
    )

    with hstrat_aux.log_context_duration("sim.run", logger=print):
        sim.run()

    phylo_df = _extract_phylo(sim.people.infection_log, variant_flavors)
    phylo_df["ancestral_sequence"] = phylo_df["variant_flavor"].map(
        reference_sequences.get,
    )
    phylo_df["sequence"] = (
        phylo_df["sequence_focal"] + phylo_df["sequence_background"]
    )

    print(f"{phylo_df['variant'].value_counts()=}")
    glimpse_df(phylo_df, logger=print)

    with hstrat_aux.log_context_duration("_add_sequence_diffs", logger=print):
        phylo_df = _add_sequence_diffs(phylo_df=phylo_df)

    fil = phylo_df["sequence_diff"].str.startswith('{"0": ')
    print(
        f"{phylo_df.loc[fil, 'sequence_diff'].str.slice(0, 10).value_counts()}"
    )
    glimpse_df(phylo_df, logger=print)

    with hstrat_aux.log_context_duration("finalize phylo_df", logger=print):
        phylo_df["py_random_sample2"] = random.getrandbits(32)
        phylo_df["np_random_sample2"] = np.random.randint(2**32)
        phylo_df["mls0_group_id"] = phylo_df["id"]
        phylo_df["mls1_group_id"] = phylo_df["id"]
        phylo_df["platform"] = "covasim"
        phylo_df["divergence_from_root"] = phylo_df["date"]

        for k, v in cfg.items():
            phylo_df[k] = v

        phylo_df = shrink_df(phylo_df, inplace=True)

    return phylo_df


if __name__ == "__main__":
    hstrat_aux.configure_prod_logging()
    cfg = read_config(sys.stdin)
    cfg["replicate_uuid"] = strong_uuid4_str()

    phylo_df = main(cfg)

    glimpse_df(phylo_df.head(), logger=print)
    glimpse_df(phylo_df.tail(), logger=print)

    assert phylo_df["sequence_diff"].str.len().fillna(0).all()

    with hstrat_aux.log_context_duration("phylo_df.to_parquet", logger=print):
        phylo_df.to_parquet(
            f"a=run_covaphastsim+replicate={cfg['replicate_uuid']}.pqt",
        )
