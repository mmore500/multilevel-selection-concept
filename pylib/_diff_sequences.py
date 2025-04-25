import typing

import numpy as np
import polars as pl


def diff_sequences(
    sequences: typing.Sequence[str],
    *,
    ancestral_sequence: str,
) -> pl.Series:

    N = len(sequences)
    L = len(ancestral_sequence)
    assert all(len(seq) == L for seq in sequences)

    # join into a flat string and reshape to 2D array
    flat = "".join(sequences).encode("ascii")
    assert len(flat) == N * L
    seq_arr = np.frombuffer(flat, dtype="S1").reshape((N, L))

    anc_flat = ancestral_sequence.encode("ascii")
    anc_arr = np.frombuffer(anc_flat, dtype="S1")

    # find mismatches
    mismatches_idxs = np.argwhere(seq_arr != anc_arr)
    mismatch_row, mismatch_pos = mismatches_idxs[:, 0], mismatches_idxs[:, 1]

    mismatch_chars = seq_arr[mismatch_row, mismatch_pos]

    # prepare df with mismatches as rows
    df = pl.DataFrame(
        {
            "index": mismatch_row,
            "pos": mismatch_pos,
            "sequence": mismatch_chars,
        },
    ).lazy()

    # prepare diff string for each position
    df = df.with_columns(
        diff=pl.concat_str(
            pl.col("pos"),
            pl.lit(': "'),
            pl.col("sequence"),
            pl.lit('"'),
        )
    )

    # aggregate each index into one comma‐separated diff string
    df = df.group_by("index").agg(diff=pl.col("diff").str.join(","))

    # prepare a base index 0..N to get one row per sequence, even if no diffs
    base = pl.select(index=pl.arange(len(sequences), dtype=pl.UInt32)).lazy()

    # left‐join the diffs, wrap in braces, fill nulls to get "{}"
    result = (
        base.join(df, how="left", on="index")
        .sort("index")
        .select(
            pl.concat_str(
                pl.lit("{"), pl.col("diff").fill_null(""), pl.lit("}")
            ),
        )
        .collect()
        .to_series()
    )

    return result
