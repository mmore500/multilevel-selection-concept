from pylib._make_cv_sim_uk import make_cv_sim_uk


def test_make_cv_sim_uk():
    sim = make_cv_sim_uk()
    sim.run()
