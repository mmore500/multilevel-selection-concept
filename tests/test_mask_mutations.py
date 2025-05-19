import numpy as np
import pytest

from pylib._mask_mutations import mask_mutations


def test_mask_mutations_basic():
    ancestral_sequence = "ACGT"
    derived_sequences = ["ACGT", "AAGT", "ACAT"]

    # Expected mutations:
    # Site 1 (ancestral 'C'): mutation with 'A'
    #   -> [False, True, False] because only the second sequence has 'A'
    # Site 2 (ancestral 'G'): mutation with 'A'
    #   -> [False, False, True] because only the third sequence has 'A'
    expected = {
        (1, "C", "A"): np.array([False, True, False]),
        (2, "G", "A"): np.array([False, False, True]),
    }

    result = mask_mutations(ancestral_sequence, derived_sequences)

    # Check that the keys (mutations) are exactly as expected.
    assert set(result.keys()) == set(expected.keys())

    # For each mutation, verify that the returned boolean arrays match.
    for key in expected:
        np.testing.assert_array_equal(result[key], expected[key])


def test_mask_mutations_invalid_length():
    ancestral_sequence = "ACGT"
    derived_sequences = ["ACGT", "ACG"]  # Second sequence is shorter

    # Test that a ValueError is raised if the lengths do not match.
    with pytest.raises(
        ValueError, match="All sequences must be of same length"
    ):
        mask_mutations(ancestral_sequence, derived_sequences)
