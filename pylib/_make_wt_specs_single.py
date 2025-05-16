import typing

from ._VariantSpec import VariantSpec


def make_wt_specs_single(
    reference_sequences: dict,
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

    return [wildtype]
