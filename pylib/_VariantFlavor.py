from dataclasses import dataclass

import covasim as cv


@dataclass
class VariantFlavor:

    variant_wt: cv.variant
    variant_mut: cv.variant

    withinhost_r_wt: float
    withinhost_r_mut: float

    active_strain_factor_wt: float
    active_strain_factor_mut: float

    p_wt_to_mut: float

    label: str
    sequence: str
