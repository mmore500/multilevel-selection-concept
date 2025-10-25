import covasim as cv
import numpy as np


def make_cv_sim_mi(
    scale_factor=0.01, start_day="2023-01-01", end_day="2023-06-30"
):
    """
    Create a Covasim simulation for contemporary COVID-19 community spread in
    Michigan.

    Currently, just starter code generated via LLM.
    """

    # ------------------------------------------------------------------------
    # 1. BASIC POPULATION + DEMOGRAPHICS
    # ------------------------------------------------------------------------
    # 'pop_size' is how many agents are in the simulation (after scaling).
    # 'pop_scale' is how to interpret results relative to the real population.
    base_michigan_population = 10_000_000
    scaled_population = int(base_michigan_population * scale_factor)

    sim_pars = {
        "pop_size": scaled_population,
        "pop_scale": scale_factor,
        "pop_type": "hybrid",  # Simple population contact network
        # synthpops can be used for more detail
        "pop_infected": 1000,  # Starting number of infected
        "beta": 0.015,  # Base transmission probability per contact
        "start_day": start_day,
        "end_day": end_day,
        "location": "usa-michigan",
        "rand_seed": 42,  # For reproducibility
    }

    # ------------------------------------------------------------------------
    # 2. VACCINATION PARAMETERS
    # ------------------------------------------------------------------------
    # We assume a certain fraction of the population is fully vaccinated
    # and a subset has received a booster. The real data vary by age/risk group;
    # here is a simplified approach.

    # For Covasim's built-in vaccine approach, we can define an intervention.
    # Example: 70% are fully vaccinated by '2023-01-01'

    vaccine = cv.vaccinate_prob(
        vaccine="pfizer",  # A built-in Covasim vaccine profile
        days=[0],  # Start giving primary vaccine on the first simulation day
        prob=0.70,  # Target 70% coverage
        subtarget=None,  # Could be an age-based, left None for simplicity
    )

    # ------------------------------------------------------------------------
    # 3. VARIANTS (OMICRON)
    # ------------------------------------------------------------------------

    omicron = cv.variant(
        "gamma",
        days=np.arange(1000),
        n_imports=100,
    )
    omicron.p["rel_beta"] = 5.8
    omicron.p["rel_severe_prob"] = 0.12

    variants = [omicron]

    # ------------------------------------------------------------------------
    # 4. PUBLIC HEALTH RESPONSES (INTERVENTIONS)
    # ------------------------------------------------------------------------
    # We'll define a moderate set of ongoing NPIs
    # (non-pharmaceutical interventions)
    #   - Some testing & isolation
    #   - vaccinations

    # Testing intervention with a certain test probability:
    # e.g. 0.02 tests/day (2% of population per day),
    # and isolation if positive for 5 days.
    testing = cv.test_prob(
        symp_prob=0.2,  # 20% of symptomatic get tested daily
        asymp_prob=0.01,  # 1% of asymptomatic get tested daily
        symp_quar_prob=0.8,  # If positive, 80% of symptomatic self-isolate
        asymp_quar_prob=0.5,  # If positive, 50% of asymptomatic self-isolate
    )

    # Consolidate all interventions:
    interventions = [
        testing,
        vaccine,
    ]

    # ------------------------------------------------------------------------
    # 5. CREATE AND RETURN THE SIM
    # ------------------------------------------------------------------------
    sim = cv.Sim(pars=sim_pars, interventions=interventions, variants=variants)

    return sim
