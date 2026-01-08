from __future__ import annotations

import pandas as pd

from ..utils.utils_time import safe_to_datetime, attach_date_to_time_series


def _find_column(df: pd.DataFrame, candidates):
    for col in df.columns:
        if col.lower() in candidates:
            return col
    return None


def _band_mean(df: pd.DataFrame, keyword: str) -> pd.Series:
    cols = [c for c in df.columns if keyword in c.lower()]
    if not cols:
        return pd.Series([float("nan")] * len(df))
    return df[cols].mean(axis=1, skipna=True)


def load_muse(path: str, session_date, log):
    df = pd.read_csv(path)
    time_col = _find_column(df, {"timestamp", "time", "datetime", "date_time"})
    if not time_col:
        log(f"Muse: missing datetime column in {path}")
        return pd.DataFrame()

    dt = safe_to_datetime(df[time_col])
    if dt.isna().all():
        log("Muse: failed to parse datetime column, no valid timestamps")
        return pd.DataFrame()

    if dt.dt.year.min() < 2000:
        dt = attach_date_to_time_series(df[time_col], session_date)

    out = pd.DataFrame({
        "datetime": dt,
        "muse_alpha_mean": _band_mean(df, "alpha"),
        "muse_beta_mean": _band_mean(df, "beta"),
        "muse_theta_mean": _band_mean(df, "theta"),
        "muse_gamma_mean": _band_mean(df, "gamma"),
    })
    out["muse_beta_alpha_ratio"] = out["muse_beta_mean"] / out["muse_alpha_mean"].replace(0, pd.NA)

    hb_col = _find_column(df, {"headbandon", "head_band_on", "headband_on"})
    if hb_col:
        out["HeadBandOn"] = df[hb_col]
    else:
        out["HeadBandOn"] = 1

    return out
