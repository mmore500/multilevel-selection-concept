from dataclasses import dataclass


@dataclass
class VariantSpec:

    sequence: str
    variant: dict
    label: str
    days: object
    n_imports: int

    withinhost_r: float = 2.0
    active_strain_factor: float = 1.0
