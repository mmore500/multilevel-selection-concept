import typing

import covasim as cv

from ._VariantFlavor import VariantFlavor


def make_flavored_variants(
    variant_flavors: typing.List[VariantFlavor],
) -> typing.List[cv.variant]:
    return [
        variant
        for flavor in variant_flavors
        for variant in (flavor.variant_wt, flavor.variant_mut)
    ]
