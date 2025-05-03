import sys
import typing

from hstrat import _auxiliary_lib as hstrat_aux
import numpy as np

from ._summarize_sequence_diffs import summarize_sequence_diffs


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

    (mut_unique, mut_counts, columns) = summarize_sequence_diffs(
        sequence_diffs=sequence_diffs,
    )

    if not len(mut_unique):
        assert len(mut_counts) == 0
        assert len(columns) == 0
        return

    print(
        f"{ancestral_sequence[0]=} ",
        f"{int(mut_unique[0])=}",
        f"{int(mut_unique[0] >> 8)=}",
        f"{chr(mut_unique[0] & 0xFF)=}",
        f"{int(mut_counts[0])=}",
        f"{int(mut_counts[0]) / len(sequence_diffs)=}",
        sep="\n",
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
