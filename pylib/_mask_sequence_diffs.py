import io
import typing

import numpy as np
import polars as pl


def mask_sequence_diffs(
    *,
    ancestral_sequence: str,
    sequence_diffs: typing.Sequence[str],
    mut_freq_thresh: int = 0,
    progress_wrap: typing.Callable = lambda x: x,
) -> typing.Iterable[typing.Tuple[typing.Tuple[int, str, str], np.ndarray]]:
    diffs = pl.DataFrame(
        {"diffs": sequence_diffs},
        schema={"diffs": pl.Utf8},
    )

    key_blob = (
        diffs.lazy()
        .select(
            pl.col("diffs")
            .filter(pl.col("diffs") != "{}")
            .str.head(-1)
            .str.tail(-1)
            .str.join(",")
        )
        .collect()
        .item()
    )
    key_list = " ".join(key_blob.replace(":", ",").split(",")[::2]).replace(
        '"', ""
    )

    if key_list.strip():
        mut_counts = np.unique_counts(
            np.loadtxt(io.StringIO(key_list), dtype=int)
        )
    else:  # avoid numpy warning
        mut_counts = np.unique_counts(np.array([], dtype=int))

    frequent_muts = mut_counts.values[mut_counts.counts >= mut_freq_thresh]
    frequent_muts.sort()

    for pos in progress_wrap(frequent_muts):
        vals = diffs["diffs"].str.json_path_match(f"$.{pos}")
        assert vals.count()
        for char in sorted(vals.drop_nulls().unique()):
            assert char != ancestral_sequence[pos]
            assert isinstance(char, str) and len(char) == 1
            ancestral_char = ancestral_sequence[pos]
            assert isinstance(ancestral_char, str) and len(ancestral_char) == 1
            yield (
                (int(pos), ancestral_char, char),
                (vals == char).fill_null(False).to_numpy(),
            )
