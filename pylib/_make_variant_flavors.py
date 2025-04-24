import typing

import covasim as cv

from ._VariantFlavor import VariantFlavor
from ._VariantSpec import VariantSpec


def make_variant_flavors(
    wt_specs: typing.Iterable[VariantSpec],
    *,
    mut_variant: typing.Callable[[VariantSpec], dict] = lambda x: x.variant,
    mut_withinhost_r: typing.Callable[
        [VariantSpec], float
    ] = lambda x: x.withinhost_r,
    mut_active_strain_factor: typing.Callable[
        [VariantSpec], float
    ] = lambda x: x.active_strain_factor,
    p_wt_to_mut: typing.Callable[[VariantSpec], float] = lambda __: 0.05,
    suffix_mut="'",
    suffix_wt="+",
) -> typing.List[VariantFlavor]:
    return [
        VariantFlavor(
            variant_wt=cv.variant(
                variant=spec.variant,
                label=f"{spec.label}{suffix_wt}",
                days=spec.days,
                n_imports=spec.n_imports,
            ),
            variant_mut=cv.variant(
                variant=mut_variant(spec),
                label=f"{spec.label}{suffix_mut}",
                days=0,
                n_imports=0,
            ),
            withinhost_r_wt=spec.withinhost_r,
            withinhost_r_mut=mut_withinhost_r(spec),
            active_strain_factor_wt=spec.active_strain_factor,
            active_strain_factor_mut=mut_active_strain_factor(spec),
            p_wt_to_mut=p_wt_to_mut(spec),
            sequence=spec.sequence,
            label=spec.label,
        )
        for spec in wt_specs
    ]
