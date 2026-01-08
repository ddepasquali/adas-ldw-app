from __future__ import annotations

from typing import Dict, Any

import json
import numpy as np
import pandas as pd

from ..io.io_muse import load_muse
from ..io.io_polar import load_polar


MUSE_FEATURES = [
    "muse_alpha_mean",
    "muse_beta_mean",
    "muse_theta_mean",
    "muse_gamma_mean",
    "muse_beta_alpha_ratio",
]

POLAR_FEATURES = [
    "polar_hr",
    "polar_rmssd",
    "polar_sdnn",
    "polar_lfhf",
]


def _stats(series: pd.Series):
    values = series.dropna().astype(float)
    if values.empty:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "n": int(values.shape[0]),
    }


def compute_baseline(
    muse_path: str,
    polar_path: str,
    session_date,
    polar_quality_thr: float,
    out_path: str,
    log,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "muse_baseline_file": muse_path,
        "polar_baseline_file": polar_path,
        "muse": {},
        "polar": {},
    }

    if muse_path:
        muse_df = load_muse(muse_path, session_date, log)
        if not muse_df.empty:
            reliable = muse_df.get("HeadBandOn", 1) == 1
            for feat in MUSE_FEATURES:
                summary["muse"][feat] = _stats(muse_df.loc[reliable, feat])
        else:
            log("Baseline: Muse baseline empty or unreadable")

    if polar_path:
        polar_df = load_polar(polar_path, session_date, log)
        if not polar_df.empty:
            if "polar_quality_pct" in polar_df.columns:
                reliable = polar_df["polar_quality_pct"] >= polar_quality_thr
            else:
                reliable = pd.Series([True] * len(polar_df))
            for feat in POLAR_FEATURES:
                if feat in polar_df.columns:
                    summary["polar"][feat] = _stats(polar_df.loc[reliable, feat])
        else:
            log("Baseline: Polar baseline empty or unreadable")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _apply_z(series: pd.Series, stats: Dict[str, Any]) -> pd.Series:
    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None or std is None or std == 0:
        return pd.Series([float("nan")] * len(series))
    return (series.astype(float) - mean) / std


def apply_baseline_zscores(
    muse_df: pd.DataFrame,
    polar_df: pd.DataFrame,
    baseline: Dict[str, Any],
    polar_quality_thr: float,
):
    if muse_df is not None and not muse_df.empty:
        muse_df = muse_df.copy()
        muse_reliable = muse_df.get("HeadBandOn", 1) == 1
        muse_df["muse_reliable"] = muse_reliable.astype(int)
        for feat in MUSE_FEATURES:
            if feat in muse_df.columns and feat in baseline.get("muse", {}):
                z = _apply_z(muse_df[feat], baseline["muse"][feat])
                z = z.where(muse_reliable, other=np.nan)
                muse_df[f"{feat}_z"] = z
    if polar_df is not None and not polar_df.empty:
        polar_df = polar_df.copy()
        if "polar_quality_pct" in polar_df.columns:
            polar_reliable = polar_df["polar_quality_pct"] >= polar_quality_thr
        else:
            polar_reliable = pd.Series([True] * len(polar_df))
        polar_df["polar_reliable"] = polar_reliable.astype(int)
        for feat in POLAR_FEATURES:
            if feat in polar_df.columns and feat in baseline.get("polar", {}):
                z = _apply_z(polar_df[feat], baseline["polar"][feat])
                z = z.where(polar_reliable, other=np.nan)
                polar_df[f"{feat}_z"] = z
    return muse_df, polar_df
