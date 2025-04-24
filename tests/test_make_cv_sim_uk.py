import os

import pandas as pd

from pylib._make_cv_sim_uk import make_cv_sim_uk
from pylib._make_flavored_variants import make_flavored_variants
from pylib._make_variant_flavors import make_variant_flavors
from pylib._make_wt_specs_uk import make_wt_specs_uk

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_make_cv_sim_uk():
    reference_sequences = pd.read_csv(f"{assets}/alignedsequences.csv")
    sequence_lookup = dict(
        zip(
            reference_sequences["WHO Label"].values,
            # remove whitespace pollution
            # and only use first 100 characters of the sequence, for perf/memory
            reference_sequences["Aligned Sequence"]
            .str.replace(r"\s+", "", regex=True)
            .str.slice(0, 100)
            .values,
        ),
    )

    wt_specs = make_wt_specs_uk(reference_sequences=sequence_lookup)
    flavored_variants = make_variant_flavors(wt_specs)
    variants = make_flavored_variants(flavored_variants)

    sim = make_cv_sim_uk(variants=variants, pop_size=10000)
    sim.run()
