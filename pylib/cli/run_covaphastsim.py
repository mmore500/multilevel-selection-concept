from collections import defaultdict
import pprint
import sys
import typing
import uuid

import covasim as cv
from hstrat import _auxiliary_lib as hstrat_aux
import pandas as pd

from .._SyncHostCompartments import SyncHostCompartments
from .._VariantFlavor import VariantFlavor
from .._cv_infection_log_to_alstd_df import cv_infection_log_to_alstd_df
from .._generate_dummy_sequences_phastSim import (
    generate_dummy_sequences_phastSim,
)
from .._make_cv_sim_uk import make_cv_sim_uk
from .._make_flavored_variants import make_flavored_variants
from .._make_variant_flavors import make_variant_flavors
from .._make_wt_specs_uk import make_wt_specs_uk
from .._read_config import read_config
from .._seed_global_rngs import seed_global_rngs
from .._shrink_df import shrink_df


def _get_reference_sequences(
    cfg: dict,
) -> typing.Dict[str, str]:
    reference_sequences = pd.read_csv("https://osf.io/hp25c/download")
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
        p_wt_to_mut=lambda __: cfg["cfg_p_wt_to_mut"],
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

    with hstrat_aux.log_context_duration(
        "generate_dummy_sequences_phastSim", logger=print
    ):
        seq_df = generate_dummy_sequences_phastSim(
            phylo_df,
            ancestral_sequences=reference_sequences,
        )

    with hstrat_aux.log_context_duration("extract variant", logger=print):
        seq_df["variant"] = seq_df["id"].map(
            phylo_df.set_index("id")["variant"].to_dict(),
        )

    with hstrat_aux.log_context_duration("prepend sequence", logger=print):
        seq_df["sequence"] = (
            seq_df["variant"]
            .str.contains(cfg["cfg_suffix_mut"])
            .map(
                {
                    True: cfg["cfg_suffix_mut"],
                    False: cfg["cfg_suffix_wt"],
                },
            )
            + seq_df["sequence"]
        )

    return seq_df


if __name__ == "__main__":

    cfg = read_config(sys.stdin)
    cfg["replicate_uuid"] = str(uuid.uuid4())
    pprint.PrettyPrinter(depth=4).pprint(cfg)
    seed_global_rngs(cfg["trt_seed"])

    reference_sequences = _get_reference_sequences(cfg)
    sim, variant_flavors = _setup_sim(
        cfg, reference_sequences=reference_sequences
    )

    with hstrat_aux.log_context_duration("sim.run", logger=print):
        sim.run()

    phylo_df = _extract_phylo(sim.people.infection_log, variant_flavors)
    seq_df = _generate_sequences(
        phylo_df,
        cfg=cfg,
        reference_sequences=reference_sequences,
    )

    with hstrat_aux.log_context_duration("phylo_df.merge", logger=print):
        phylo_df = phylo_df.reset_index(drop=True).merge(
            seq_df.reset_index(drop=True).drop(
                [col for col in phylo_df.columns if col != "id"],
                axis="columns",
                errors="ignore",
            ),
            on="id",
        )

    with hstrat_aux.log_context_duration("finalize phylo_df", logger=print):
        for k, v in cfg.items():
            phylo_df[k] = v

        phylo_df = shrink_df(phylo_df, inplace=True)

    print(phylo_df.head())

    with hstrat_aux.log_context_duration("phylo_df.to_parquet", logger=print):
        phylo_df.to_parquet(
            f"a=run_covaphastsim+replicate={cfg['replicate_uuid']}.pqt",
        )
