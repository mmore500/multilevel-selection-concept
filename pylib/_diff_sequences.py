import typing

from hstrat import _auxiliary_lib as hstrat_aux
import numpy as np
import polars as pl


def _diff_sequences_batch(
    sequences: typing.Sequence[str],
    *,
    ancestral_sequence: str,
    quote_keys: bool = True,
) -> pl.Series:

    N = len(sequences)
    L = len(ancestral_sequence)
    assert all(len(seq) == L for seq in sequences)

    # join into a flat string and reshape to 2D array
    with hstrat_aux.log_context_duration("seq_arr", logger=print):
        flat = "".join(sequences).encode("ascii")
        assert len(flat) == N * L
        seq_arr = np.frombuffer(flat, dtype="S1").reshape((N, L))

    with hstrat_aux.log_context_duration("anc_arr", logger=print):
        anc_flat = ancestral_sequence.encode("ascii")
        anc_arr = np.frombuffer(anc_flat, dtype="S1")

    # find mismatches
    with hstrat_aux.log_context_duration("mismatches", logger=print):
        mismatches_idxs = np.argwhere(seq_arr != anc_arr)
        mismatch_row, mismatch_pos = (
            mismatches_idxs[:, 0],
            mismatches_idxs[:, 1],
        )

        mismatch_chars = seq_arr[mismatch_row, mismatch_pos]

        # prepare df with mismatches as rows
        df_ = pl.DataFrame(
            {
                "index": mismatch_row,
                "pos": mismatch_pos,
                "sequence": mismatch_chars,
            },
        )

    df = df_.lazy()

    # prepare diff string for each position
    key_quote = '"' * bool(quote_keys)
    df = df.with_columns(
        diff=pl.concat_str(
            pl.lit(key_quote),
            pl.col("pos"),
            pl.lit(f'{key_quote}: "'),
            pl.col("sequence"),
            pl.lit('"'),
        )
    )

    # aggregate each index into one comma‐separated diff string
    df = df.group_by("index").agg(diff=pl.col("diff").str.join(", "))

    # prepare a base index 0..N to get one row per sequence, even if no diffs
    index_dtype = df_["index"].dtype
    base = pl.select(index=pl.arange(N, dtype=index_dtype)).lazy()

    # left‐join the diffs, wrap in braces, fill nulls to get "{}"
    res = (
        base.join(df, how="left", on="index")
        .sort("index")
        .select(
            pl.concat_str(
                pl.lit("{"), pl.col("diff").fill_null(""), pl.lit("}")
            ),
        )
    )

    with hstrat_aux.log_context_duration("res.collect", logger=print):
        return res.collect().to_series()


def diff_sequences(
    sequences: typing.Sequence[str],
    *,
    ancestral_sequence: str,
    batch_size: int = 20_000,
    progress_wrap: typing.Callable = lambda x: x,
    quote_keys: bool = True,
) -> pl.Series:
    """
    Compare sequences to an ancestral sequence and return a diff string.

    Parameters
    ----------
    sequences : typing.Sequence[str]
        The sequences to compare.
    ancestral_sequence : str
        The ancestral sequence.
    batch_size : int, default 20_000
        The number of sequences to process in each batch.
    progress_wrap : typing.Callable, default lambda x: x
        Pass tqdm or equivalent to display progress bar.
    quote_keys : bool, default True
        Whether to quote the keys in the diff string.

    Returns
    -------
    pl.Series
        A series of diff strings for each sequence.
    """
    # batch in order to avoid memory exposion
    return pl.concat(
        [
            _diff_sequences_batch(
                sequences[i : i + batch_size],
                ancestral_sequence=ancestral_sequence,
                quote_keys=quote_keys,
            )
            for i in progress_wrap(range(0, len(sequences), batch_size))
        ],
    )
