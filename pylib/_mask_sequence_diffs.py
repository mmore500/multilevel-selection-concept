import io
import typing

import numpy as np
import polars as pl


def mask_sequence_diffs(
    *,
    ancestral_sequence: str,
    sequence_diffs: typing.Sequence[str],
    mut_count_thresh: int = 0,
    mut_quart_thresh: float = 0.0,
    progress_wrap: typing.Callable = lambda x: x,
) -> typing.Iterable[typing.Tuple[typing.Tuple[int, str, str], np.ndarray]]:
    diffs = pl.DataFrame({"diffs": sequence_diffs}, schema={"diffs": pl.Utf8})

    mut_tokens = (
        diffs.lazy()
        .drop_nulls()
        .filter(pl.col("diffs") != "{}")
        .filter(pl.col("diffs") != "")
        .select(
            pl.col("diffs").str.head(-1).str.tail(-1).str.join(","),
        )
        .collect()
        .item()
        .replace(":", ",")
        .replace('"', "")
        .replace(",", " ")
        .split()
    )

    if not mut_tokens:
        return

    pos_vals = np.loadtxt(io.StringIO(" ".join(mut_tokens[::2])), dtype=int)
    char_vals = np.array(mut_tokens[1::2], dtype="S1").view(np.uint8)
    mut_uids = pos_vals.astype(np.uint64) << 8 | char_vals

    (mut_unique, mut_counts) = np.unique(mut_uids, return_counts=True)
    print(
        f"{ancestral_sequence[0]=} "
        f"{int(mut_unique[0])=} "
        f"{int(mut_unique[0] >> 8)=} "
        f"{chr(mut_unique[0] & 0xFF)=} "
        f"{int(mut_counts[0])=}",
    )

    mut_count_thresh = max(
        mut_count_thresh,
        np.quantile(mut_counts, mut_quart_thresh),
    )
    is_frequent_mut = mut_counts >= mut_count_thresh

    seq_diff_sizes = diffs["diffs"].str.count_matches(":").fill_null(0)
    seq_diff_rows = np.repeat(np.arange(len(seq_diff_sizes)), seq_diff_sizes)

    for mut_uid in progress_wrap(mut_unique[is_frequent_mut]):
        mask = np.zeros(len(sequence_diffs), dtype=bool)
        mask[seq_diff_rows[mut_uid == mut_uids]] = True

        pos, mut_char_var = int(mut_uid >> 8), chr(mut_uid & 0xFF)
        mut_char_ref = ancestral_sequence[pos]

        assert mut_char_var != ancestral_sequence[pos]
        assert isinstance(mut_char_var, str) and len(mut_char_var) == 1
        assert isinstance(mut_char_ref, str) and len(mut_char_ref) == 1
        yield (pos, mut_char_ref, mut_char_var), mask
