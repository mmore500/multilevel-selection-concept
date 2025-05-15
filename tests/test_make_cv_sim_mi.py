from pylib._make_cv_sim_mi import make_cv_sim_mi


def test_make_cv_sim_mi():
    sim = make_cv_sim_mi()
    sim.run()
