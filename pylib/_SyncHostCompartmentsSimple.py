import warnings

import covasim as cv
import more_itertools as mit
import numpy as np

from ._VariantFlavor import VariantFlavor


def _elapse_day(
    compartments: np.ndarray,
    flavor: VariantFlavor,
    host_capacity: float,
) -> np.ndarray:
    f = flavor
    num_doublings = int(np.floor(np.log2(f.withinhost_r_wt)))
    if not num_doublings > 0:
        raise ValueError

    p_per_doubling = 1.0 - np.power(1.0 - f.p_wt_to_mut, 1 / num_doublings)

    wt_growth_per_doubling = f.withinhost_r_wt ** (1 / num_doublings)
    mut_growth_per_doubling = f.withinhost_r_mut ** (1 / num_doublings)
    assert (
        abs(wt_growth_per_doubling**num_doublings - f.withinhost_r_wt) < 1e-6
    )
    assert (
        abs(mut_growth_per_doubling**num_doublings - f.withinhost_r_mut)
        < 1e-6
    )

    offset = 0

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
                host_capacity,
            )
            / host_capacity
        )

    return compartments


def _bootstrap_transition_probabilities(
    flavor: VariantFlavor,
    host_capacity: float,
    num_days: int,
    init: int,
    n_bootstrap: int = 100_000,
) -> list:
    """Sample a bootstrap distribution of transition probabilities."""
    compartments = np.zeros((n_bootstrap, 2), dtype=float)
    compartments[:, init] = 1.0

    res = []
    for __ in range(num_days):
        compartments_ = compartments.copy()
        compartments_ *= np.random.rand(*compartments.shape)

        offset = 0
        compartments_[:, offset] *= flavor.active_strain_factor_wt
        compartments_[:, offset + 1] *= flavor.active_strain_factor_mut

        sampled_strains = np.argmax(compartments_, axis=1)
        res.append((sampled_strains != init).mean())
        compartments = _elapse_day(
            compartments,
            flavor=flavor,
            host_capacity=host_capacity,
        )

    return res


class SyncHostCompartmentsSimple:

    _transition_probabilities: dict[str, list[float]]
    _variant_flavors: list[VariantFlavor]
    _infection_log_pos: int

    _infection_days: np.ndarray

    def __init__(
        self: "SyncHostCompartmentsSimple",
        *,
        pop_size: int,
        variant_flavors: list[VariantFlavor],
        # see https://doi.org/10.1073/pnas.2024815118
        host_capacity: float = 1e10,
    ) -> None:

        self._variant_flavors = variant_flavors
        self._infection_log_pos = 0
        self._flavor_positions = {
            flavor.label: i * 2 + 1 for i, flavor in enumerate(variant_flavors)
        }
        self._0to1_transition_probabilities = {
            flavor.label: _bootstrap_transition_probabilities(
                flavor=flavor,
                host_capacity=host_capacity,
                num_days=100,
                init=0,
            )
            for flavor in variant_flavors
        }
        self._1to0_transition_probabilities = {
            flavor.label: _bootstrap_transition_probabilities(
                flavor=flavor,
                host_capacity=host_capacity,
                num_days=100,
                init=1,
            )
            for flavor in variant_flavors
        }

        self._infection_days = np.zeros(pop_size, dtype=int)

    def __call__(self: "SyncHostCompartmentsSimple", sim: cv.Sim) -> None:
        people = sim.people
        log = people.infection_log

        for entry in log[self._infection_log_pos :]:
            source, target, variant = (
                entry["source"],
                entry["target"],
                entry["variant"],
            )

            self._infection_days[target] = entry["date"]
            if source is None:
                continue

            flavor = mit.one(
                f.label
                for f in self._variant_flavors
                if variant.startswith(f.label)
            )
            elapsed_days = entry["date"] - self._infection_days[source]

            if np.isnan(people["exposed_variant"][target]) and np.isnan(
                people["infectious_variant"][target]
            ):
                warnings.warn(
                    "exposed_variant and infectious_variant are both NaN"
                )
                continue

            if variant.endswith("+"):
                lookup = self._0to1_transition_probabilities
                transition_p = lookup[flavor][elapsed_days]
                assert people["exposed_variant"][target] % 2 == 1
                delta = np.random.rand() < transition_p
                people["exposed_variant"][target] += delta
                people["infectious_variant"][target] += delta
                entry["variant"] = variant[:-1] + "'"
            elif variant.endswith("'"):
                lookup = self._1to0_transition_probabilities
                transition_p = lookup[flavor][elapsed_days]
                assert people["exposed_variant"][target] % 2 == 0
                delta = np.random.rand() < transition_p
                people["exposed_variant"][target] -= delta
                people["infectious_variant"][target] -= delta
                entry["variant"] = variant[:-1] + "+"
            else:
                raise ValueError("Unsupported variant suffix")

            if not (
                np.isnan(people["infectious_variant"][target])
                or np.isnan(people["exposed_variant"][target])
            ):
                people["infectious_variant"][target] = people[
                    "exposed_variant"
                ][target]

        self._infection_log_pos = len(log)
