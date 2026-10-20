"""
qrmlib.missing -- describing which cells of a return table are observed.

A blank cell in the CSV files means "did not trade that day" and is read by
pandas as NaN.  Everything in this module works on a pandas DataFrame whose
columns are series and whose rows are days.
"""
from __future__ import annotations

import pandas as pd


def observed_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean table, True where the series traded (cell is not NaN)."""
    return df.notna()


def joint_obs_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Number of days on which each *pair* of series is jointly observed.

    With O the 0/1 observed matrix (days x series), O' O has entry (i, j) equal
    to the number of days on which both i and j are observed.  The diagonal is
    simply the number of observed days of each series.
    """
    obs = df.notna().astype(int)
    return obs.T @ obs


def complete_rows(df: pd.DataFrame) -> pd.Series:
    """Boolean Series, True on days where *every* series is observed."""
    return df.notna().all(axis=1)


def n_complete(df: pd.DataFrame) -> int:
    """Number of days on which every series is observed."""
    return int(complete_rows(df).sum())


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-series count of observed / missing days and percent missing."""
    n = len(df)
    observed = df.notna().sum()
    return pd.DataFrame({
        "observed": observed,
        "missing": n - observed,
        "pct_missing": 100.0 * (n - observed) / n,
    })
