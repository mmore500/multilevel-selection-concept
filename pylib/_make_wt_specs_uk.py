# adapted from
# https://github.com/Jasminapg/Covid-19-Analysis/blob/7a26db7b5f985e09a4af30e6d6605a570535940b/8_omicron_analysis/calibrate_uk.py

from datetime import date
import typing

import numpy as np

from ._VariantSpec import VariantSpec


def make_wt_specs_uk(
    reference_sequences: dict,
    start_day: str = "2020-01-20",
) -> typing.List[VariantSpec]:

    baseline_variant_params = {
        "rel_beta": 1.0,
        "rel_symp_prob": 1.0,
        "rel_severe_prob": 1.0,
        "rel_crit_prob": 1.0,
        "rel_death_prob": 1.0,
    }

    wildtype = VariantSpec(
        sequence=reference_sequences["Wildtype"],
        variant=baseline_variant_params,
        label="Wildtype",
        days=[0],
        n_imports=1000,
    )

    # adding different variants: B.1.177 in September 2020, Alpha slightly
    # later and Delta from April 2021
    start_day = date(*map(int, start_day.split("-")))

    # # Add B.1.177 strain from September 2020 and assume it's like b1351 (no
    # # vaccine at this time in England)
    # variants = []
    # b1177 = cv.variant(
    #     "b1351",
    #     days=np.arange(sim.day("2020-08-10"), sim.day("2020-08-20")),
    #     n_imports=3000,
    # )
    # b1177.p["rel_beta"] = 1.2
    # b1177.p["rel_severe_prob"] = 0.4
    # variants += [b1177]
    first_day = (date(*map(int, "2020-08-10".split("-"))) - start_day).days
    last_day = (date(*map(int, "2020-08-20".split("-"))) - start_day).days
    b1351 = VariantSpec(
        sequence=reference_sequences["Beta"],
        variant={
            **baseline_variant_params,
            "rel_beta": 1.2,
            "rel_severe_prob": 0.4,
        },
        label="Beta",
        days=np.arange(first_day, last_day),
        n_imports=3000,
    )

    # # Add Alpha strain from October 2020
    # b117 = cv.variant(
    #     "b117",
    #     days=np.arange(sim.day("2020-10-20"), sim.day("2020-10-30")),
    #     n_imports=3000,
    # )
    # b117.p["rel_beta"] = 1.8
    # b117.p["rel_severe_prob"] = 0.4
    # variants += [b117]
    first_day = (date(*map(int, "2020-10-20".split("-"))) - start_day).days
    last_day = (date(*map(int, "2020-10-30".split("-"))) - start_day).days
    b117 = VariantSpec(
        sequence=reference_sequences["Alpha"],
        variant={
            **baseline_variant_params,
            "rel_beta": 1.8,
            "rel_severe_prob": 0.4,
        },
        label="Alpha",
        days=np.arange(first_day, last_day),
        n_imports=3000,
    )

    # # Add Delta strain starting middle of April
    # b16172 = cv.variant(
    #     "b16172",
    #     days=np.arange(sim.day("2021-04-15"), sim.day("2021-04-20")),
    #     n_imports=4000,
    # )
    # b16172.p["rel_beta"] = 3.1
    # b16172.p["rel_severe_prob"] = 0.2
    # variants += [b16172]
    first_day = (date(*map(int, "2021-04-15".split("-"))) - start_day).days
    last_day = (date(*map(int, "2021-04-20".split("-"))) - start_day).days
    b16172 = VariantSpec(
        sequence=reference_sequences["Delta"],
        variant={
            **baseline_variant_params,
            "rel_beta": 3.1,
            "rel_severe_prob": 0.2,
        },
        label="Delta",
        days=np.arange(first_day, last_day),
        n_imports=4000,
    )

    return [wildtype, b1351, b117, b16172]
