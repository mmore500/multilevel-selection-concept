from dataclasses import dataclass


@dataclass
class VariantSpec:

    sequence: str
    variant: dict
    label: str
    days: object
    n_imports: int

    # https://doi.org/10.1038/s43856-022-00195-4
    # assume doubling time of four hours
    withinhost_r: float = 2**6
    active_strain_factor: float = 1.0
