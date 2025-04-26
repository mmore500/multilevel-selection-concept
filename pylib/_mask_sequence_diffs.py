import io
import typing

import numpy as np
import polars as pl


def mask_sequence_diffs(
    *,
    ancestral_sequence: str,
    sequence_diffs: typing.Sequence[str],
) -> typing.Iterable[typing.Tuple[typing.Tuple[int, str], np.ndarray]]:
    diffs = pl.DataFrame(
        {"diffs": sequence_diffs},
        schema={"diffs": pl.Utf8},
    )

    key_blob = (
        diffs.lazy()
        .select(pl.col("diffs").str.head(-1).str.tail(-1).str.join(","))
        .collect()
        .item()
    )
    key_list = " ".join(key_blob.replace(":", ",").split(",")[::2]).replace(
        '"', ""
    )
    keys = np.unique(np.loadtxt(io.StringIO(key_list), dtype=int))
    keys.sort()

    for pos in keys:
        vals = diffs["diffs"].str.json_path_match(f"$.{pos}")
        assert vals.count()
        for char in sorted(vals.unique()):
            assert char != ancestral_sequence[pos]
            yield (pos, char), (vals == char).to_numpy()
