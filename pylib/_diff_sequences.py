import typing

import polars as pl


def diff_sequences(
    sequences: typing.Sequence[str],
    *,
    ancestral_sequence: str,
) -> pl.Series:
    # build initial dataframe
    df = pl.DataFrame(
        {
            "sequence": sequences,
            "ancestral_sequence": ancestral_sequence,
        },
    )
    assert (len(ancestral_sequence) == df["sequence"].str.len_bytes()).all()

    df = df.lazy()

    # explode each charcter from sequence
    df = (
        df.with_row_index()
        .with_columns(
            ancestral_sequence=pl.col("ancestral_sequence").str.split(""),
            sequence=pl.col("sequence").str.split(""),
        )
        .explode("ancestral_sequence", "sequence")
        .with_columns(pos=pl.cum_count("index") % len(ancestral_sequence))
    )

    # only consider mismatches
    df = df.filter(pl.col("ancestral_sequence") != pl.col("sequence"))

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
    df_diffs = df.group_by("index").agg(
        diff=pl.col("diff").str.join(","),
    )

    # prepare a base index 0..N to get one row per sequence, even if no diffs
    base = pl.select(index=pl.arange(len(sequences), dtype=pl.UInt32)).lazy()

    # left‐join the diffs, wrap in braces, fill nulls to get "{}"
    result = (
        base.join(df_diffs, how="left", on="index")
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
