from collections import defaultdict
import pprint
import random
import sys
import typing

import covasim as cv
from hstrat import _auxiliary_lib as hstrat_aux
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import tmap as tqdm_tmap

from .._SyncHostCompartments import SyncHostCompartments
from .._VariantFlavor import VariantFlavor
from .._cv_infection_log_to_alstd_df import cv_infection_log_to_alstd_df
from .._diff_sequences import diff_sequences
from .._generate_dummy_sequences_phastSim import (
    generate_dummy_sequences_phastSim,
)
from .._glimpse_df import glimpse_df
from .._make_cv_sim_uk import make_cv_sim_uk
from .._make_flavored_variants import make_flavored_variants
from .._make_variant_flavors import make_variant_flavors
from .._make_wt_specs_uk import make_wt_specs_uk
from .._read_config import read_config
from .._seed_global_rngs import seed_global_rngs
from .._shrink_df import shrink_df
from .._shuffle_string import shuffle_string
from .._strong_uuid4_str import strong_uuid4_str


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

    wt_specs = make_wt_specs_uk(reference_sequences=reference_sequences)
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

    return (
        make_cv_sim_uk(
            preinterventions=[
                SyncHostCompartments(
                    variant_flavors=variant_flavors,
                    pop_size=cfg["cfg_pop_size"],
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
        phylo_df = cv_infection_log_to_alstd_df(infection_log)

    with hstrat_aux.log_context_duration("alifestd_join_roots", logger=print):
        phylo_df = hstrat_aux.alifestd_join_roots(phylo_df, mutate=True)

    with hstrat_aux.log_context_duration(
        "alifestd_to_working_format", logger=print
    ):
        phylo_df = hstrat_aux.alifestd_to_working_format(phylo_df, mutate=True)

    with hstrat_aux.log_context_duration("map variant_flavor", logger=print):
        phylo_df["variant_flavor"] = phylo_df["variant"].map(
            {
                v.label: vf.label
                for vf in variant_flavors
                for v in (vf.variant_mut, vf.variant_wt)
            },
        )

    return phylo_df


def _generate_sequences(
    phylo_df: pd.DataFrame,
    *,
    cfg: typing.Dict,
    reference_sequences: typing.Dict[str, str],
) -> pd.DataFrame:

    # workaround to generate sequences for all nodes, not just leaves
    dummy_leaves = phylo_df.copy()
    dummy_leaves["ancestor_id"] = dummy_leaves["id"]
    id_delta = phylo_df["id"].max() + 1
    dummy_leaves["id"] += id_delta

    with hstrat_aux.log_context_duration(
        "generate_dummy_sequences_phastSim", logger=print
    ):
        seq_df = generate_dummy_sequences_phastSim(
            pd.concat([phylo_df, dummy_leaves], ignore_index=True),
            ancestral_sequences=reference_sequences,
            progress_map=tqdm_tmap,
        )
        seq_df["id"] -= id_delta  # revert dummy leaves back to true nodes

    with hstrat_aux.log_context_duration("extract variant", logger=print):
        seq_df["variant"] = seq_df["id"].map(
            phylo_df.set_index("id")["variant"].to_dict(),
        )

    with hstrat_aux.log_context_duration("prepend sequence", logger=print):

        suffix = cfg["cfg_suffix_wt"] * (cfg["cfg_num_mut_sites"] - 1)

        seq_df["sequence"] = (
            seq_df["variant"]
            .str.contains(cfg["cfg_suffix_mut"])
            .map(
                {
                    True: cfg["cfg_suffix_mut"] + suffix,
                    False: cfg["cfg_suffix_wt"] + suffix,
                },
            )
            .apply(shuffle_string)
            + seq_df["sequence"]
        )

    assert len(seq_df) == len(phylo_df)
    return seq_df


def _add_sequence_diffs(phylo_df: pd.DataFrame):
    assert hstrat_aux.alifestd_is_topologically_sorted(phylo_df)
    assert hstrat_aux.alifestd_count_root_nodes(phylo_df) == 1

    phylo_df = phylo_df.set_index("id", drop=False)
    ancestral_sequence = phylo_df.at[0, "sequence"]
    phylo_df["ancestral_sequence"] = ancestral_sequence
    assert phylo_df["sequence"].str.len().nunique() == 1

    phylo_df["sequence_diff"] = diff_sequences(
        phylo_df["sequence"],
        ancestral_sequence=ancestral_sequence,
        progress_wrap=tqdm,
    )
    del phylo_df["sequence"]

    return phylo_df


if __name__ == "__main__":

    cfg = read_config(sys.stdin)
    cfg["replicate_uuid"] = strong_uuid4_str()
    pprint.PrettyPrinter(depth=4).pprint(cfg)
    seed_global_rngs(cfg["trt_seed"])
    cfg["py_random_state1"] = str(random.getstate())
    cfg["np_random_state1"] = str(np.random.get_state())
    cfg["py_random_sample1"] = random.getrandbits(32)
    cfg["np_random_sample1"] = np.random.randint(2**32)

    reference_sequences = _get_reference_sequences(cfg)
    sim, variant_flavors = _setup_sim(
        cfg, reference_sequences=reference_sequences
    )

    with hstrat_aux.log_context_duration("sim.run", logger=print):
        sim.run()

    phylo_df = _extract_phylo(sim.people.infection_log, variant_flavors)
    print(f"{phylo_df['variant'].value_counts()=}")
    glimpse_df(phylo_df, logger=print)

    seq_df = _generate_sequences(
        phylo_df,
        cfg=cfg,
        reference_sequences=reference_sequences,
    )

    glimpse_df(seq_df, logger=print)

    with hstrat_aux.log_context_duration("phylo_df.merge", logger=print):
        phylo_df = phylo_df.reset_index(drop=True).merge(
            seq_df.reset_index(drop=True).drop(
                [col for col in phylo_df.columns if col != "id"],
                axis="columns",
                errors="ignore",
            ),
            on="id",
        )

    with hstrat_aux.log_context_duration("_add_sequence_diffs", logger=print):
        phylo_df = _add_sequence_diffs(phylo_df=phylo_df)

    fil = phylo_df["sequence_diff"].str.startswith('{"0": ')
    print(
        f"{phylo_df.loc[fil, 'sequence_diff'].str.slice(0, 10).value_counts()}"
    )
    glimpse_df(seq_df, logger=print)

    with hstrat_aux.log_context_duration("finalize phylo_df", logger=print):
        phylo_df["py_random_state2"] = str(random.getstate())
        phylo_df["np_random_state2"] = str(np.random.get_state())
        phylo_df["py_random_sample2"] = random.getrandbits(32)
        phylo_df["np_random_sample2"] = np.random.randint(2**32)
        phylo_df["mls0_group_id"] = phylo_df["id"]
        phylo_df["mls1_group_id"] = phylo_df["id"]
        phylo_df["platform"] = "covaphast"
        phylo_df["divergence_from_root"] = phylo_df["date"]

        for k, v in cfg.items():
            phylo_df[k] = v

        phylo_df = shrink_df(phylo_df, inplace=True)

    glimpse_df(phylo_df.head(), logger=print)
    glimpse_df(phylo_df.tail(), logger=print)

    with hstrat_aux.log_context_duration("phylo_df.to_parquet", logger=print):
        phylo_df.to_parquet(
            f"a=run_covaphastsim+replicate={cfg['replicate_uuid']}.pqt",
        )
