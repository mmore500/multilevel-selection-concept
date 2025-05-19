import numbers

import pandas as pd


def shrink_df(
    df: pd.DataFrame, *, inplace: bool = False, nunique_thresh: int = 10
) -> pd.DataFrame:

    df = df if inplace else df.copy()

    for col in df.select_dtypes(
        include=["object"],
    ).columns:
        if df[col].nunique() <= nunique_thresh:
            df[col] = df[col].astype("category")

    # adapted from https://stackoverflow.com/a/67403354
    for col in df.columns:
        # integers
        if issubclass(df[col].dtypes.type, numbers.Integral):
            # unsigned integers
            if df[col].min() >= 0:
                df[col] = pd.to_numeric(df[col], downcast="unsigned")
            # signed integers
            else:
                df[col] = pd.to_numeric(df[col], downcast="integer")
        # other real numbers
        elif issubclass(df[col].dtypes.type, numbers.Real):
            df[col] = pd.to_numeric(df[col], downcast="float")

    return df
