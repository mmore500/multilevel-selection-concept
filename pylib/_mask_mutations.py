import typing

import numpy as np


def mask_mutations(
    ancestral_sequence: str,
    derived_sequences: typing.Sequence[str],
) -> typing.Dict[tuple, np.ndarray]:
    """
    Generate boolean masks for mutations in derived sequences relative to an
    ancestral sequence.

    Assumes all sequences are of the same length.

    Parameters
    ----------
    ancestral_sequence : str
        The ancestral sequence.
    derived_sequences : sequence of str
        Derived sequences, each the same length as the ancestral sequence.

    Returns
    -------
    dict of (int, str, str) -> np.ndarray
        Dictionary mapping each (position, ancestral, derived) to a boolean
        array indicating which derived sequences contain the mutation.
    """
    if not all(
        len(seq) == len(ancestral_sequence) for seq in derived_sequences
    ):
        raise ValueError(
            "All sequences must be of same length as ancestral sequence.",
        )

    ancestral_bytes = np.frombuffer(
        ancestral_sequence.encode("utf-8"), dtype="S1"
    )
    arr = np.frombuffer("".join(derived_sequences).encode("utf-8"), dtype="S1")
    arr = arr.reshape(len(derived_sequences), len(ancestral_sequence))

    # Return a dictionary mapping (site, mutation) to a boolean mask array.
    return {
        (site, ancestral_sequence[site], character.decode("utf-8")): arr[
            :, site
        ]
        == character
        for site in range(len(ancestral_sequence))
        for character in np.unique(arr[:, site])
        if character != ancestral_bytes[site]
    }
