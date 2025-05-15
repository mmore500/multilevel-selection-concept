from hstrat import _auxiliary_lib as hstrat_aux
import pandas as pd


def cv_infection_log_to_alstd_df(
    log_records: list,
    join_roots: bool = True,
) -> pd.DataFrame:
    df = pd.DataFrame(log_records).sort_values("date").reset_index(drop=True)
    df.loc[df["source"].isna(), "source"] = df.loc[
        df["source"].isna(), "target"
    ]
    df["source"] = df["source"].astype(int)

    most_recent_event = {}  # person_id -> row number
    ancestor_ids = []
    for i, row in df.iterrows():
        source, target = row["source"], row["target"]
        ancestor_ids.append(most_recent_event.get(source, i))
        most_recent_event[target] = i

    df["id"] = range(len(df))
    df["ancestor_id"] = ancestor_ids
    df["origin_time"] = df["date"]

    df["ancestor_list"] = hstrat_aux.alifestd_make_ancestor_list_col(
        df["id"],
        df["ancestor_id"],
    )
    if join_roots:
        df = hstrat_aux.alifestd_join_roots(df)
        df["ancestor_list"] = hstrat_aux.alifestd_make_ancestor_list_col(
            df["id"],
            df["ancestor_id"],
        )

    return df
