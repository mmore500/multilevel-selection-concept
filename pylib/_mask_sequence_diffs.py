import io
import typing

import numpy as np
import polars as pl


def mask_sequence_diffs(
    *,
    ancestral_sequence: str,
    sequence_diffs: typing.Sequence[str],
    progress_wrap: typing.Callable = lambda x: x,
) -> typing.Iterable[typing.Tuple[typing.Tuple[int, str, str], np.ndarray]]:
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
    if key_list.strip():
        keys = np.unique(np.loadtxt(io.StringIO(key_list), dtype=int))
    else:
        keys = np.array([], dtype=int)  # avoid numpy warning

    keys.sort()

    for pos in progress_wrap(keys):
        vals = diffs["diffs"].str.json_path_match(f"$.{pos}")
        assert vals.count()
        for char in sorted(vals.unique()):
            assert char != ancestral_sequence[pos]
            yield (pos, ancestral_sequence[pos], char), (
                vals == char
            ).to_numpy()
