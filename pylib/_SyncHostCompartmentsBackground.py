import covasim as cv
import numpy as np

from ._VariantFlavor import VariantFlavor


class SyncHostCompartmentsBackground:

    _host_capacity: float
    _host_compartments: np.ndarray
    _variant_flavors: list[VariantFlavor]
    _infection_log_pos: int
    _infectious_variants_log: list[np.ndarray]
    _infection_log_entries: list[dict]
    _infection_days_elapsed: np.ndarray
    _last_sampled_strains: np.ndarray

    def __init__(
        self: "SyncHostCompartmentsBackground",
        *,
        pop_size: int,
        variant_flavors: list[VariantFlavor],
        num_background_strains: int,
        # see https://doi.org/10.1073/pnas.2024815118
        host_capacity: float = 1e10,
    ) -> None:
        # overall wildtype,
        # then mut/wildtype per flavor
        num_variants = len(variant_flavors) * 2 + 1
        shape = (pop_size, num_variants, num_background_strains)
        self._host_capacity = host_capacity
        self._host_compartments = np.zeros(shape, dtype=float)
        self._variant_flavors = variant_flavors
        self._infection_log_pos = 0
        self._infectious_variants_log = []
        self._infection_log_entries = [None] * pop_size
        self._infection_days_elapsed = np.zeros(pop_size, dtype=int)
        self._last_sampled_strains = np.zeros(
            (pop_size, num_background_strains), dtype=int
        )

    def __call__(self: "SyncHostCompartmentsBackground", sim: cv.Sim) -> None:
        compartments = self._host_compartments
        people = sim.people
        log = people.infection_log
        var_lookup = {v: k for k, v in sim["variant_map"].items()}
        num_variants = self._host_compartments.shape[1]
        assert len(var_lookup) == num_variants

        self._infection_days_elapsed += self._infection_days_elapsed > 0

        ## sync covasim to host compartments
        #######################################################################
        for entry in log[self._infection_log_pos :]:
            source, target, variant, date = (
                entry["source"],
                entry["target"],
                entry["variant"],
                entry["date"],
            )
            self._infection_log_entries[target] = entry
            self._infection_days_elapsed[target] = 1

            if source is not None:
                variant = self._infectious_variants_log[date][source].astype(
                    int
                )
                assert len(variant) == compartments.shape[2]
            else:
                variant = [var_lookup[variant]] * compartments.shape[2]

            entry["sequence_background"] = "".join(
                ["'", "+"][v % 2] for v in variant
            )

            # zero out non-infectious/exposed compartments
            compartments[target, :, :] = 0.0

            # init w/ covasim infectious variant
            compartments[target, variant, :] = 1.0

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
            assert (
                abs(
                    wt_growth_per_doubling**num_doublings
                    - variant_flavor.withinhost_r_wt
                )
                < 1e-6
            )
            for __ in range(num_doublings):
                compartments[:, offset, :] *= wt_growth_per_doubling
                compartments[:, offset + 1, :] *= wt_growth_per_doubling

                pop_wt = compartments[:, offset, :].astype(int)
                pop_mt = compartments[:, offset + 1, :].astype(int)
                mask_wt = pop_wt > 0
                mask_mt = pop_mt > 0
                num_mutants = np.zeros_like(pop_wt)
                num_reversions = np.zeros_like(pop_mt)
                num_mutants[mask_wt] = np.random.binomial(
                    pop_wt[mask_wt], p_per_doubling
                )
                num_reversions[mask_mt] = np.random.binomial(
                    pop_mt[mask_mt], p_per_doubling
                )

                compartments[:, offset, :] -= num_mutants
                compartments[:, offset + 1, :] += num_mutants
                compartments[:, offset, :] += num_reversions
                compartments[:, offset + 1, :] -= num_reversions

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
            compartments_[
                :, offset, :
            ] *= variant_flavor.active_strain_factor_wt
            compartments_[
                :, offset + 1, :
            ] *= variant_flavor.active_strain_factor_mut

        sampled_strains = np.where(
            compartments_.any(axis=1),
            np.argmax(compartments_, axis=1),
            np.nan,
        )
        assert len(sampled_strains) == compartments_.shape[0]

        self._last_sampled_strains = np.where(
            ~np.isnan(sampled_strains),
            sampled_strains,
            self._last_sampled_strains,
        )

        # update current covasim infectious variant
        self._infectious_variants_log.append(sampled_strains)

        ## sample variants of record
        for who in np.flatnonzero(self._infection_days_elapsed >= 8):
            entry = self._infection_log_entries[who]
            assert entry is not None
            assert not (self._last_sampled_strains[who] == 0).any()
            variant = self._last_sampled_strains[who].astype(int)
            entry["sequence_background"] = "".join(
                ["'", "+"][v % 2] for v in variant
            )
            self._infection_log_entries[who] = None

        self._infection_days_elapsed *= self._infection_days_elapsed < 8

        # zero out non-infectious/exposed compartments
        compartments[sim.people.recovered | sim.people.dead, :, :] = 0.0
