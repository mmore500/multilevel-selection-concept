import covasim as cv
import numpy as np

from ._VariantFlavor import VariantFlavor


class SyncHostCompartments:

    _host_capacity: float
    _host_compartments: np.ndarray
    _variant_flavors: list[VariantFlavor]
    _infection_log_pos: int

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
        self._infection_log_pos = 0

    def __call__(self: "SyncHostCompartments", sim: cv.Sim) -> None:
        compartments = self._host_compartments
        people = sim.people
        log = people.infection_log
        var_lookup = {v: k for k, v in sim["variant_map"].items()}
        num_variants = self._host_compartments.shape[1]
        assert len(var_lookup) == num_variants

        ## sync covasim to host compartments
        #######################################################################
        for entry in log[self._infection_log_pos :]:
            _source, target, variant = (
                entry["source"],
                entry["target"],
                entry["variant"],
            )
            # zero out non-infectious/exposed compartments
            compartments[target, :] = 0.0

            variant = var_lookup[variant]

            # init w/ covasim infectious variant
            compartments[target, variant] = 1.0

            entry["sequence_focal"] = ["'", "+"][variant % 2]

        self._infection_log_pos = len(log)

        assert (
            not (people["infectious_variant"] <= 0).any()
            and not (people["infectious_variant"] >= num_variants).any()
        )

        # update host compartments using luria-delbruck dynamics
        #######################################################################
        for i, variant_flavor in enumerate(self._variant_flavors):
            offset = i * 2 + 1

            num_doublings = int(
                np.floor(np.log2(variant_flavor.withinhost_r_wt))
            )
            if not num_doublings > 0:
                raise ValueError
            p_per_doubling = 1.0 - np.power(
                1.0 - variant_flavor.p_wt_to_mut, 1 / num_doublings
            )

            wt_growth_per_doubling = variant_flavor.withinhost_r_wt ** (
                1 / num_doublings
            )
            mut_growth_per_doubling = variant_flavor.withinhost_r_mut ** (
                1 / num_doublings
            )
            assert (
                abs(
                    wt_growth_per_doubling**num_doublings
                    - variant_flavor.withinhost_r_wt
                )
                < 1e-6
            )
            assert (
                abs(
                    mut_growth_per_doubling**num_doublings
                    - variant_flavor.withinhost_r_mut
                )
                < 1e-6
            )
            for __ in range(num_doublings):
                compartments[:, offset] *= wt_growth_per_doubling
                compartments[:, offset + 1] *= mut_growth_per_doubling
                num_mutants = np.random.binomial(
                    compartments[:, offset].astype(int),
                    p_per_doubling,
                )
                num_reversions = np.random.binomial(
                    compartments[:, offset + 1].astype(int),
                    p_per_doubling,
                )
                compartments[:, offset] -= num_mutants
                compartments[:, offset + 1] += num_mutants
                compartments[:, offset] += num_reversions
                compartments[:, offset + 1] -= num_reversions

        # apply within-host carrying capacity
        compartments /= (
            np.maximum(
                compartments.sum(axis=1, keepdims=True),
                self._host_capacity,
            )
            / self._host_capacity
        )

        ## sync host compartments to covasim "infectious variant"
        #######################################################################
        # sample current infectious variant from compartments
        compartments_ = compartments.copy()
        np.nan_to_num(compartments_, copy=False)
        assert np.isfinite(compartments_).all()
        compartments_ *= np.random.rand(*compartments.shape)

        for i, variant_flavor in enumerate(self._variant_flavors):
            offset = i * 2 + 1
            compartments_[:, offset] *= variant_flavor.active_strain_factor_wt
            compartments_[
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
