"""Shared fixtures for the churn prediction test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal DataFrame with representative columns across all cleaning/feature steps."""
    return pd.DataFrame(
        {
            # ── Leakage ──
            "churndep": [1, 0, 1, 0],
            # ── Billing / usage (median-imputed) ──
            "revenue": [100.0, 200.0, np.nan, 300.0],
            "mou": [500.0, np.nan, 600.0, 700.0],
            "recchrge": [50.0, 60.0, np.nan, 80.0],
            "directas": [10.0, 20.0, 30.0, np.nan],
            "overage": [5.0, np.nan, 15.0, 20.0],
            "roam": [0.0, 1.0, 0.0, np.nan],
            # ── Change metrics ──
            "changem": [np.nan, 0.5, 1.0, 0.0],
            "changer": [0.1, np.nan, 0.3, 0.0],
            # ── Device ──
            "phones": [1, 2, np.nan, 1],
            "models": [2020, 2021, 2022, np.nan],
            "eqpdays": [100, np.nan, 200, 150],
            # ── Age ──
            "age1": [35, np.nan, 45, 25],
            "age2": [np.nan, 30, 50, 20],
            # ── Drop / call columns ──
            "dropvce": [1, 0, 2, 1],
            "peakvce": [100, 200, 150, 180],
            "opeakvce": [50, 60, 70, 40],
            "outcalls": [30, 40, 20, 50],
            "incalls": [70, 60, 80, 50],
            "custcare": [1, 2, 0, 3],
            "retcalls": [0, 1, 0, 0],
            "months": [12, 24, 6, 36],
            # ── Binary features ──
            "refurb": [0, 1, 0, 1],
            "webcap": [1, 0, 1, 1],
            "newcelly": [0, 0, 1, 0],
            "newcelln": [1, 1, 0, 1],
            "truck": [0, 0, 0, 0],
            "rv": [0, 0, 0, 0],
            "pcown": [1, 0, 1, 0],
            "ownrent": [1, 1, 0, 0],
            "mcycle": [0, 0, 0, 0],
            "occprof": [1, 0, 0, 0],
            "occcler": [0, 1, 0, 0],
            "occcrft": [0, 0, 1, 0],
            "occstud": [0, 0, 0, 1],
            "occhmkr": [0, 0, 0, 0],
            "occret": [0, 0, 0, 0],
            "occself": [0, 0, 0, 0],
            "marryyes": [1, 0, 1, 0],
            "marryun": [0, 0, 0, 1],
            "prizmrur": [0, 1, 0, 0],
            "prizmub": [0, 0, 1, 1],
            "prizmtwn": [1, 0, 1, 0],
            "mailord": [0, 0, 0, 1],
            "mailres": [1, 1, 0, 0],
            "mailflag": [1, 0, 1, 0],
            "travel": [0, 1, 0, 0],
            "creditcd": [1, 0, 1, 0],
            "retcall": [0, 0, 1, 0],
            "retaccpt": [1, 1, 0, 0],
            "credita": [0, 0, 0, 0],
            "creditaa": [1, 1, 1, 1],
            "children": [0, 2, 1, 0],
            "incmiss": [0, 0, 0, 0],
            "setprcm": [0, 0, 0, 0],
            # ── Target ──
            "churn": [1, 0, 1, 0],
        }
    )


@pytest.fixture
def sample_df_clean(sample_df: pd.DataFrame) -> pd.DataFrame:
    """A pre-cleaned version (as if clean_data already ran)."""
    df = sample_df.copy()
    # Drop leakage
    df = df.drop(columns=["churndep"], errors="ignore")
    # Median imputation (manual for reproducible fixture)
    for col in df.columns:
        if df[col].dtype in ("float64", "int64") and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    # Binary columns to int
    from src.config import Columns
    for col in Columns.BINARY_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df
