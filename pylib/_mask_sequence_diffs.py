import typing

import numpy as np
import polars as pl


def mask_sequence_diffs(
    *,
    ancestral_sequence: str,
    sequence_diffs: typing.Sequence[str],
) -> typing.Iterable[typing.Tuple[typing.Tuple[int, str], np.ndarray]]:
    diffs = pl.Series(sequence_diffs).cast(pl.Utf8)
    for pos in range(len(ancestral_sequence)):
        vals = diffs.str.json_path_match(f"$.{pos}")
        if vals.count() == 0:
            continue
        for char in sorted(vals.unique()):
            assert char != ancestral_sequence[pos]
            yield (pos, char), (vals == char).to_numpy()
