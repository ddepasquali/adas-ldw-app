from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def make_time_grid(max_t: float, hz: float) -> np.ndarray:
    if hz <= 0:
        hz = 10.0
    step = 1.0 / hz
    return np.arange(0.0, max_t + step / 2, step)


def resample_df(df: pd.DataFrame, t_grid: np.ndarray) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame({"t_seconds": t_grid})
    df = df.copy()
    if "t_seconds" not in df.columns:
        raise ValueError("Expected t_seconds column in dataframe")
    keep_cols = ["t_seconds"] + [
        c for c in df.select_dtypes(include=["number", "bool"]).columns if c != "t_seconds"
    ]
    df = df[keep_cols]
    df = df.sort_values("t_seconds").drop_duplicates("t_seconds")
    df = df.set_index("t_seconds")
    df = df.reindex(t_grid, method="nearest", tolerance=(t_grid[1] - t_grid[0]))
    df = df.interpolate(method="linear", limit_direction="both")
    df = df.reset_index().rename(columns={"index": "t_seconds"})
    return df


def align_and_resample(
    sources: Dict[str, pd.DataFrame],
    resample_hz: float,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    max_t = 0.0
    for df in sources.values():
        if df is None or df.empty:
            continue
        max_t = max(max_t, float(df["t_seconds"].max()))
    t_grid = make_time_grid(max_t, resample_hz)

    resampled: Dict[str, pd.DataFrame] = {}
    for name, df in sources.items():
        resampled[name] = resample_df(df, t_grid)

    fused = pd.DataFrame({"t_seconds": t_grid})
    for name, df in resampled.items():
        if df.empty:
            continue
        cols = [c for c in df.columns if c != "t_seconds"]
        renamed = {c: f"{name}_{c}" for c in cols}
        fused = fused.merge(df.rename(columns=renamed), on="t_seconds", how="left")

    return resampled, fused
