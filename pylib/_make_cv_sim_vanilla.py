import typing

import covasim as cv


def make_cv_sim_vanilla(
    *,
    preinterventions: typing.Sequence[object] = tuple(),
    postinterventions: typing.Sequence[object] = tuple(),
    pop_size: int = 100_000,
    seed: int = 1,
    variants: typing.List[cv.variant],
) -> cv.Sim:

    sim = cv.Sim(
        interventions=[
            *preinterventions,
            *postinterventions,
        ],
        n_days=650,
        use_waning=False,
        pop_infected=0,  # disable wild-type strain
        pop_size=pop_size,
        variants=variants,
        rand_seed=seed,
    )

    sim.initialize()

    return sim
