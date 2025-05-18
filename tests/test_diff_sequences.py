from pylib._diff_sequences import diff_sequences


def test_diff_sequences():
    """Test the diff_sequences function with a simple example."""
    sequences = ["ACGT", "ACGT", "ACGT", "XCGX"]
    ancestral_sequence = "ACGT"
    assert diff_sequences(
        sequences,
        ancestral_sequence=ancestral_sequence,
        quote_keys=False,
    ).to_list() == ["{}", "{}", "{}", '{0: "X", 3: "X"}']
    assert diff_sequences(
        sequences,
        ancestral_sequence=ancestral_sequence,
        quote_keys=True,
    ).to_list() == ["{}", "{}", "{}", '{"0": "X", "3": "X"}']
