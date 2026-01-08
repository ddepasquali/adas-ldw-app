from __future__ import annotations

from datetime import datetime, date, time, timedelta
from typing import Iterable, Optional

import pandas as pd


_TIME_FORMATS = [
    "%H:%M:%S.%f",
    "%H:%M:%S,%f",
    "%H:%M:%S",
]


def parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def parse_time_of_day(time_str: str) -> time:
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(time_str, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Unsupported time format: {time_str}")


def combine_date_time(session_date: date, time_str: str) -> datetime:
    t = parse_time_of_day(time_str)
    return datetime.combine(session_date, t)


def attach_date_to_time_series(
    series: pd.Series, session_date: date
) -> pd.Series:
    return series.apply(lambda s: datetime.combine(session_date, parse_time_of_day(str(s))))


def add_t_seconds(df: pd.DataFrame, global_start: datetime) -> pd.DataFrame:
    df = df.copy()
    df["t_seconds"] = (df["datetime"] - global_start).dt.total_seconds()
    return df


def earliest_datetime(dfs: Iterable[pd.DataFrame], extra: Optional[Iterable[datetime]] = None) -> datetime:
    candidates = []
    for df in dfs:
        if df is None or df.empty:
            continue
        if "datetime" in df.columns:
            candidates.append(df["datetime"].min())
    if extra:
        candidates.extend([dt for dt in extra if dt is not None])
    if not candidates:
        raise ValueError("No valid datetime values found to compute global start.")
    return min(candidates)


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")
