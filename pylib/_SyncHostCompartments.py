import covasim as cv
import numpy as np

from ._VariantFlavor import VariantFlavor


class SyncHostCompartments:

    _host_capacity: float
    _host_compartments: np.ndarray
    _variant_flavors: list[VariantFlavor]

    def __init__(
        self: "SyncHostCompartments",
        *,
        pop_size: int,
        variant_flavors: list[VariantFlavor],
        # see https://doi.org/10.1073/pnas.2024815118
        host_capacity: float = 1e10,
    ) -> None:
        # overall wildtype,
        # then mut/wildtype per flavor
        num_variants = len(variant_flavors) * 2 + 1
        shape = (pop_size, num_variants)
        self._host_capacity = host_capacity
        self._host_compartments = np.zeros(shape, dtype=float)
        self._variant_flavors = variant_flavors

    def __call__(self: "SyncHostCompartments", sim: cv.Sim) -> None:
        compartments = self._host_compartments
        people = sim.people
        random_p = np.random.rand(*people["infectious_variant"].shape)

        ## sync covasim to host compartments
        #######################################################################
        # zero out non-infectious/exposed compartments
        mask = ~(people["infectious"] | people["exposed"])
        compartments[mask, :] = 0.0

        # ensure host compartments are initialized w/ covasim infectious variant
        num_variants = self._host_compartments.shape[1]
        for variant in range(1, num_variants):
            compartments[:, variant] = np.maximum(
                people["infectious_variant"] == variant,
                compartments[:, variant],
            )

        # update host compartments
        #######################################################################
        # grow strains
        for i, variant_flavor in enumerate(self._variant_flavors):
            offset = i * 2 + 1
            compartments[:, offset] *= variant_flavor.withinhost_r_wt
            compartments[:, offset + 1] *= variant_flavor.withinhost_r_mut

        # apply within-host carrying capacity
        compartments /= (
            np.maximum(
                compartments.sum(axis=1, keepdims=True),
                self._host_capacity,
            )
            / self._host_capacity
        )

        # introduce low-transmissibility variants thru spontaneous mutation
        # of high-transmissibility variants
        # e.g., gamma -> gamma' and delta -> delta'
        for i, variant_flavor in enumerate(self._variant_flavors):
            offset = i * 2 + 1
            p_unit = 1.0 - variant_flavor.p_wt_to_mut
            p = 1.0 - np.power(p_unit, compartments[:, offset])
            compartments[:, offset + 1] = np.maximum(
                random_p < p,
                compartments[:, offset + 1],
            )

        ## sync host compartments to covasim "infectious variant"
        #######################################################################
        # sample current infectious variant from compartments
        compartments_ = compartments.copy()
        compartments_ *= np.random.rand(*compartments.shape)

        for i, variant_flavor in enumerate(self._variant_flavors):
            offset = i * 2 + 1
            compartments[:, offset] *= variant_flavor.active_strain_factor_wt
            compartments[
                :, offset + 1
            ] *= variant_flavor.active_strain_factor_mut

        sampled_strains = np.where(
            compartments_.any(axis=1),
            np.argmax(compartments_, axis=1),
            np.nan,
        )

        # update current covasim infectious variant
        people["infectious_variant"] = np.where(
            ~np.isnan(people["infectious_variant"]),
            sampled_strains,
            people["infectious_variant"],
        )
