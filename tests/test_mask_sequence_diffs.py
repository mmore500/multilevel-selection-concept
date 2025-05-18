import numpy as np
import pytest

from pylib._mask_sequence_diffs import mask_sequence_diffs


def test_mask_sequence_diffs_simple():
    ancestral_sequence = "ACG"
    # Two JSON‐encoded diff strings, none of whose values
    # equal the ancestral base at the same position.
    sequence_diffs = [
        '{"0": "T", "1": "G", "2": "A"}',
        '{"0": "T", "1": "T", "2": "A"}',
    ]

    # Collect all ( (pos, char), mask_array ) tuples
    result = list(
        mask_sequence_diffs(
            ancestral_sequence=ancestral_sequence,
            sequence_diffs=sequence_diffs,
        )
    )

    # We expect four masks:
    #  - pos=0, char='T' -> [True, True]
    #  - pos=1, char='G' -> [True, False]
    #  - pos=1, char='T' -> [False, True]
    #  - pos=2, char='A' -> [True, True]
    expected = [
        ((0, "A", "T"), np.array([True, True])),
        ((1, "C", "G"), np.array([True, False])),
        ((1, "C", "T"), np.array([False, True])),
        ((2, "G", "A"), np.array([True, True])),
    ]

    assert len(result) == len(expected)
    for (pos_char, mask), (exp_pos_char, exp_mask) in zip(result, expected):
        # correct (position, character) tuple
        assert pos_char == exp_pos_char
        # mask is a numpy array
        assert isinstance(mask, np.ndarray)
        # and its contents match exactly
        assert np.array_equal(mask, exp_mask)


def test_mask_sequence_diffs_raises_on_ancestral_conflict():
    ancestral_sequence = "AC"
    # At position 0 the diff equals the ancestral base 'A',
    # so the assert in the function should trigger.
    sequence_diffs = ['{"0":"A","1":"T"}']

    with pytest.raises(AssertionError):
        # forcing evaluation of the generator
        list(
            mask_sequence_diffs(
                ancestral_sequence=ancestral_sequence,
                sequence_diffs=sequence_diffs,
            )
        )


def test_mask_sequence_diffs_empty_input():
    ancestral_sequence = "A"
    sequence_diffs = []  # no diffs at all

    result = list(
        mask_sequence_diffs(
            ancestral_sequence=ancestral_sequence,
            sequence_diffs=sequence_diffs,
        )
    )
    # nothing to mask, should be empty
    assert result == []
