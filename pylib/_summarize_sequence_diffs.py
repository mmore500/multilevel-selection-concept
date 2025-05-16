import io
import typing

from hstrat import _auxiliary_lib as hstrat_aux
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix


def summarize_sequence_diffs(
    *,
    sequence_diffs: typing.Sequence[str],
) -> typing.Tuple[np.ndarray, np.ndarray, typing.List[np.ndarray],]:
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
        return np.array([]), np.array([]), []

    pos_vals = np.loadtxt(io.StringIO(" ".join(mut_tokens[::2])), dtype=int)
    char_vals = np.array(mut_tokens[1::2], dtype="S1").view(np.uint8)
    mut_uids = pos_vals.astype(np.uint64) << 8 | char_vals

    (mut_unique, mut_inverse, mut_counts) = np.unique(
        mut_uids, return_inverse=True, return_counts=True
    )

    with hstrat_aux.log_context_duration("seq_diff_rows", logger=print):
        seq_diff_sizes = diffs["diffs"].str.count_matches(":").fill_null(0)
        seq_diff_rows = np.repeat(
            np.arange(len(seq_diff_sizes)), seq_diff_sizes
        )
        assert len(seq_diff_rows) == len(mut_uids)

    # construct from three arrays:
    # - data[:] the entries of the matrix, in any order
    # - i[:] the row indices of the matrix entries
    # - j[:] the column indices of the matrix entries
    with hstrat_aux.log_context_duration("coo_matrix", logger=print):
        values = np.ones_like(seq_diff_rows, dtype=bool)
        row_indices = seq_diff_rows
        col_indices = mut_inverse
        coo = coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(sequence_diffs), mut_unique.size),
        )

    with hstrat_aux.log_context_duration("coo.tocsc", logger=print):
        csc = coo.tocsc()

    # adapted from https://scicomp.stackexchange.com/a/35243
    with hstrat_aux.log_context_duration("np.split", logger=print):
        boundaries = csc.indptr[1:-1]
        columns = np.split(csc.indices, boundaries)
        assert len(columns) == len(mut_unique)

    return mut_unique, mut_counts, columns
