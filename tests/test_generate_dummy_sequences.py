import pandas as pd

from pylib._generate_dummy_sequences import generate_dummy_sequences


def dummy_mutator(ancestor_sequence, variant, ancestor_variant):
    return f"{ancestor_sequence}_{variant}"


def test_generate_dummy_sequences_smoke():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 0, 1],
            "variant": ["A", "B", "C", "D"],
        }
    )

    ancestral_sequence = "seq0"

    # Expected generated sequences:
    # - Row 0 (root): returns the ancestral_sequence → "seq0".
    # - Row 1: applies dummy_mutator(seq0, "B", "A") → "seq0_B".
    # - Row 2: applies dummy_mutator(seq0, "C", "A") → "seq0_C".
    # - Row 3: applies dummy_mutator("seq0_B", "D", "B") → "seq0_B_D".
    expected_sequences = ["seq0", "seq0_B", "seq0_C", "seq0_B_D"]

    result = generate_dummy_sequences(
        phylogeny_df, ancestral_sequence, dummy_mutator
    )

    assert result == expected_sequences
