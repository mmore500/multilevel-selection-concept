import numpy as np
import pandas as pd

from pylib._run_phastSim import run_phastSim


def test_run_phastSim():

    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 0],
            "origin_time": [0, 1, 2, 3],
        }
    )
    res = run_phastSim(
        ancestral_sequence="ACGT" * 25,
        phylogeny_df=phylogeny_df,
    )

    assert "id" in res.columns
    assert "sequence" in res.columns
    assert set(res["id"]) == {2, 3}
    assert np.all(res["sequence"].str.len() == 100)
