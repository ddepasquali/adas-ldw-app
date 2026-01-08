from __future__ import annotations

import math
import pandas as pd

from ..utils.utils_time import attach_date_to_time_series


def _find_time_column(df: pd.DataFrame):
    for col in df.columns:
        if col.lower() in {"time", "timestamp", "datetime", "date_time"}:
            return col
    return None


def _find_yaw_column(df: pd.DataFrame):
    keywords = ["yaw", "gyro_z", "gyroz", "rotation_rate_z", "rot_z", "angular_velocity_z"]
    for col in df.columns:
        lc = col.lower()
        if any(kw in lc for kw in keywords):
            return col
    return None


def _to_rad_per_s(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return series
    max_val = series.abs().max()
    if max_val > 6.5:
        return series * (math.pi / 180.0)
    return series


def load_phyphox(path: str, session_date, log):
    df = pd.read_csv(path)
    time_col = _find_time_column(df)
    if not time_col:
        log(f"Phyphox: missing time column in {path}")
        return pd.DataFrame()

    dt = attach_date_to_time_series(df[time_col], session_date)
    out = pd.DataFrame({"datetime": dt})

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if col == time_col:
            continue
        out[f"phy_{col}"] = df[col]

    yaw_col = _find_yaw_column(df)
    if yaw_col:
        out["phy_yaw_rate_rads"] = _to_rad_per_s(df[yaw_col])

    return out
