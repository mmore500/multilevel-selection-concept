import typing

import pandas as pd
import polars as pl


def glimpse_df(df: pd.DataFrame, logger: typing.Callable) -> None:
    """
    Print a summary of the DataFrame using Polars.

    Args:
        df (pd.DataFrame): The DataFrame to summarize.
        logger (typing.Callable): A logging function to log the summary.
    """
    df = pl.from_pandas(df.head())
    message = df.glimpse(return_as_string=True)
    message = "\n".join(line[:150] for line in message.splitlines())
    logger(message)
