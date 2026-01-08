from __future__ import annotations

import pandas as pd

from ..utils.utils_time import attach_date_to_time_series


def _find_time_column(df: pd.DataFrame):
    for col in df.columns:
        if col.lower() in {"time", "timestamp", "datetime", "date_time"}:
            return col
    return None


def load_obd(path: str, session_date, log):
    df = pd.read_csv(path)
    time_col = _find_time_column(df)
    if not time_col:
        log(f"OBD: missing time column in {path}")
        return pd.DataFrame()

    dt = attach_date_to_time_series(df[time_col], session_date)
    out = pd.DataFrame({"datetime": dt})

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if col == time_col:
            continue
        out[f"obd_{col}"] = df[col]

    return out
