import io
import sys
import typing

from hstrat import _auxiliary_lib as hstrat_aux
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix


def mask_sequence_diffs(
    *,
    ancestral_sequence: str,
    sequence_diffs: typing.Sequence[str],
    mut_count_thresh: typing.Tuple[int, int] = (0, sys.maxsize),
    mut_freq_thresh: typing.Tuple[float, float] = (0.0, 1.0),
    mut_quant_thresh: typing.Tuple[float, float] = (0.0, 1.0),
    progress_wrap: typing.Callable = lambda x: x,
    sparsify_mask: bool = False,
) -> typing.Iterable[typing.Tuple[typing.Tuple[int, str, str], np.ndarray]]:
    if not (
        len(mut_freq_thresh) == 2
        and len(mut_count_thresh) == 2
        and len(mut_quant_thresh) == 2
    ):
        raise ValueError

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

    (mut_unique, mut_inverse, mut_counts) = np.unique(
        mut_uids, return_inverse=True, return_counts=True
    )
    print(
        f"{ancestral_sequence[0]=} "
        f"{int(mut_unique[0])=} "
        f"{int(mut_unique[0] >> 8)=} "
        f"{chr(mut_unique[0] & 0xFF)=} "
        f"{int(mut_counts[0])=}",
    )

    with hstrat_aux.log_context_duration("is_valid_mut", logger=print):
        mut_freq = mut_counts / len(sequence_diffs)
        is_valid_mut = (np.clip(mut_freq, *mut_freq_thresh) == mut_freq) & (
            np.clip(mut_counts, *mut_count_thresh) == mut_counts
        )
        print(f"{is_valid_mut[0]=}")
        print(f"{(mut_counts[is_valid_mut] < mut_counts[0]).mean()=}")

        mut_quant_thresh = tuple(
            np.quantile(mut_counts[is_valid_mut], mut_quant_thresh),
        )
        assert len(mut_quant_thresh) == 2
        is_valid_mut = is_valid_mut & (
            np.clip(mut_counts, *mut_quant_thresh) == mut_counts
        )

    print(f"{len(is_valid_mut)=} {is_valid_mut.sum()=} {is_valid_mut[0]=}")

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

    with hstrat_aux.log_context_duration("indices", logger=print):
        indices = np.flatnonzero(is_valid_mut)

    for idx in progress_wrap(indices):
        if not sparsify_mask:
            mask = np.zeros(len(sequence_diffs), dtype=bool)
            mask[columns[idx]] = True
        else:
            mask = columns[idx]

        mut_uid = mut_unique[idx]
        pos, mut_char_var = int(mut_uid >> 8), chr(mut_uid & 0xFF)
        mut_char_ref = ancestral_sequence[pos]

        assert mut_char_var != ancestral_sequence[pos]
        assert isinstance(mut_char_var, str) and len(mut_char_var) == 1
        assert isinstance(mut_char_ref, str) and len(mut_char_ref) == 1
        yield (pos, mut_char_ref, mut_char_var), mask
