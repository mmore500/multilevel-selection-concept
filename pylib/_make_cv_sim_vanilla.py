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
        use_waning=True,
        pop_infected=0,  # disable wild-type strain
        pop_size=pop_size,
        variants=variants,
        rand_seed=seed,
        nab_decay=dict(
            form="nab_growth_decay",
            growth_time=21,
            decay_rate1=0.07,
            decay_time1=47,
            decay_rate2=0.02,
            decay_time2=106,
        ),
    )

    sim.initialize()

    return sim
