"""Feature engineering for the customer churn prediction pipeline.

Defines ratio, binned, and business-signal features layered on top of the
cleaned dataset. All functions are DataFrame-in / DataFrame-out so they can be
wrapped in sklearn ``FunctionTransformer`` objects.
"""

import numpy as np
import pandas as pd



def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered ratio features derived from usage and billing columns."""
    df = df.copy()

    if {"revenue", "mou"}.issubset(df.columns):
        mou = df["mou"].replace(0, np.nan)
        df["revenue_per_mou"] = df["revenue"] / mou

    if {"dropvce", "peakvce", "opeakvce"}.issubset(df.columns):
        denom = df["dropvce"] + df["peakvce"] + df["opeakvce"]
        df["drop_call_rate"] = df["dropvce"] / denom.replace(0, np.nan)

    if {"outcalls", "incalls"}.issubset(df.columns):
        denom = df["outcalls"] + df["incalls"]
        df["outcall_ratio"] = df["outcalls"] / denom.replace(0, np.nan)

    if {"custcare", "retcalls", "months"}.issubset(df.columns):
        months = df["months"].replace(0, np.nan)
        df["support_contact_rate"] = (df["custcare"] + df["retcalls"]) / months

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features combining usage, billing, and tenure."""
    df = df.copy()

    if {"revenue", "months"}.issubset(df.columns):
        df["revenue_x_tenure"] = df["revenue"] * df["months"]

    if {"mou", "eqpdays"}.issubset(df.columns):
        df["mou_x_eqpdays"] = df["mou"] * df["eqpdays"]

    if {"overage", "mou"}.issubset(df.columns):
        mou = df["mou"].replace(0, np.nan)
        df["overage_ratio"] = df["overage"] / mou

    if {"dropblk", "peakvce", "opeakvce"}.issubset(df.columns):
        denom = (df["peakvce"] + df["opeakvce"]).replace(0, np.nan)
        df["call_failure_rate"] = df["dropblk"] / denom

    return df


def add_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add RFM-style (Recency, Frequency, Monetary) normalized features."""
    df = df.copy()

    if {"revenue", "months"}.issubset(df.columns):
        months = df["months"].replace(0, np.nan)
        df["revenue_per_month"] = df["revenue"] / months

    if {"mou", "outcalls", "incalls"}.issubset(df.columns):
        total_calls = (df["outcalls"] + df["incalls"]).replace(0, np.nan)
        df["avg_call_duration"] = df["mou"] / total_calls

    return df


def add_binned_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add categorical binned features for tenure and age."""
    df = df.copy()

    if "months" in df.columns:
        df["tenure_group"] = pd.cut( # cut converts continuous vars into categorical vars using intervals
            df["months"],
            bins=[0, 6, 12, 24, 48, 72, 2**63 - 1],
            labels=["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4-6yr", "6yr+"],
        )

    if "age1" in df.columns:
        df["age_group"] = pd.cut(
            df["age1"],
            bins=[0, 25, 35, 45, 55, 65, 2**63 - 1],
            labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
        )

    return df


def add_business_flags(
    df: pd.DataFrame,
    revenue_p90: float = 100.0,
    revenue_p75: float = 65.0,
    drop_rate_p90: float = 0.05,
) -> pd.DataFrame:
    """Add boolean-style business signal flags as 0/1 integers."""
    df = df.copy()

    if "revenue" in df.columns:
        r_p90 = df["revenue"].quantile(0.9) if len(df) > 1000 else revenue_p90
        r_p75 = df["revenue"].quantile(0.75) if len(df) > 1000 else revenue_p75
        df["is_high_value"] = (df["revenue"] >= r_p90).astype(int)

        if "months" in df.columns:
            df["is_low_tenure_high_value"] = (
                (df["months"] < 12) & (df["revenue"] >= r_p75)
            ).astype(int)

    if "drop_call_rate" in df.columns:
        d_p90 = df["drop_call_rate"].quantile(0.9) if len(df) > 1000 else drop_rate_p90
        df["is_frequent_drops"] = (df["drop_call_rate"] >= d_p90).astype(int)
        df["is_high_drops"] = (df["drop_call_rate"] >= 0.1).astype(int)

    if {"overage", "revenue"}.issubset(df.columns):
        df["is_billing_shock"] = (df["overage"] >= df["revenue"] * 0.5).astype(int)

    if {"is_high_value", "months"}.issubset(df.columns):
        df["is_high_value_new"] = (
            (df["is_high_value"] == 1) & (df["months"] < 12)
        ).astype(int)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in order and return the result."""
    df = add_ratio_features(df)
    df = add_interaction_features(df)
    df = add_rfm_features(df)
    df = add_binned_features(df)
    df = add_business_flags(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    for col in ["tenure_group", "age_group"]:
        if col in df.columns and df[col].dtype.name == "category" and df[col].isna().any():
            df[col] = df[col].cat.add_categories(["Unknown"])
            df[col] = df[col].fillna("Unknown")

    return df
