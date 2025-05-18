import numpy as np
import pandas as pd
import polars as pl


def screen_mutation_defined_nodes_sequence_diff(
    phylo_df: pd.DataFrame,
    mut_char_pos: int,
    mut_char_var: str,
) -> np.ndarray:

    if not (
        "sequence_diff" in phylo_df.columns
        and phylo_df["sequence_diff"].str.len().fillna(0).all()
    ):
        raise ValueError

    sdpl = pl.from_pandas(phylo_df["sequence_diff"])
    anc_sdpl = pl.from_pandas(
        phylo_df.loc[phylo_df["ancestor_id"].values, "sequence_diff"],
    )

    return (
        (sdpl.str.json_path_match(f"$.{mut_char_pos}") == mut_char_var)
        .fill_null(False)
        .to_numpy()
    ) & (
        anc_sdpl.str.json_path_match(f"$.{mut_char_pos}") != mut_char_var
    ).fill_null(
        True
    ).to_numpy()
