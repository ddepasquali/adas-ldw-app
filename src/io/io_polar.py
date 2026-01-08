from __future__ import annotations

import pandas as pd

from ..utils.utils_time import safe_to_datetime, attach_date_to_time_series


def _find_column_exact(df: pd.DataFrame, candidates):
    for col in df.columns:
        if col.lower() in candidates:
            return col
    return None


def _find_column_contains(df: pd.DataFrame, keywords, exclude=None):
    exclude = exclude or []
    for col in df.columns:
        lc = col.lower()
        if any(kw in lc for kw in keywords) and not any(ex in lc for ex in exclude):
            return col
    return None


def load_polar(path: str, session_date, log):
    df = pd.read_csv(path)
    time_col = _find_column_exact(df, {"timestamp", "time", "datetime", "date_time"})
    if not time_col:
        log(f"Polar: missing datetime column in {path}")
        return pd.DataFrame()

    dt = safe_to_datetime(df[time_col])
    if dt.isna().all():
        log("Polar: failed to parse datetime column, no valid timestamps")
        return pd.DataFrame()

    if dt.dt.year.min() < 2000:
        dt = attach_date_to_time_series(df[time_col], session_date)

    out = pd.DataFrame({"datetime": dt})

    hr_col = _find_column_contains(df, ["hr", "heart_rate"], exclude=["hrv"])
    if hr_col:
        out["polar_hr"] = df[hr_col]

    rmssd_col = _find_column_contains(df, ["rmssd"])
    if rmssd_col:
        out["polar_rmssd"] = df[rmssd_col]

    sdnn_col = _find_column_contains(df, ["sdnn"])
    if sdnn_col:
        out["polar_sdnn"] = df[sdnn_col]

    lfhf_col = _find_column_contains(df, ["lf_hf", "lfhf", "lf/hf"])
    if lfhf_col:
        out["polar_lfhf"] = df[lfhf_col]

    qual_col = _find_column_contains(df, ["quality"])
    if qual_col:
        out["polar_quality_pct"] = df[qual_col]

    return out
