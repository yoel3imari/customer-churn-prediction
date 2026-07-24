import numpy as np
import pandas as pd
import pytest

from src.features.build import (
    add_interaction_features,
    add_rfm_features,
    add_business_flags,
    build_features,
)
from src.features.pipeline import create_preprocessing_pipeline, get_feature_names


@pytest.fixture
def sample_phase2_df():
    return pd.DataFrame(
        {
            "revenue": [50.0, 120.0, 30.0, 80.0],
            "mou": [200.0, 500.0, 100.0, 400.0],
            "months": [6, 24, 3, 18],
            "eqpdays": [100, 300, 50, 200],
            "overage": [10.0, 70.0, 0.0, 20.0],
            "dropblk": [2, 5, 0, 1],
            "peakvce": [50, 100, 20, 80],
            "opeakvce": [30, 80, 10, 50],
            "outcalls": [15, 40, 5, 30],
            "incalls": [10, 25, 5, 20],
            "dropvce": [1, 2, 0, 1],
            "custcare": [1, 3, 0, 2],
            "retcalls": [0, 1, 0, 0],
            "age1": [30, 45, 22, 55],
            "churn": [0, 1, 0, 1],
        }
    )


def test_interaction_features(sample_phase2_df):
    df = add_interaction_features(sample_phase2_df)
    assert "revenue_x_tenure" in df.columns
    assert "mou_x_eqpdays" in df.columns
    assert "overage_ratio" in df.columns
    assert "call_failure_rate" in df.columns
    assert df["revenue_x_tenure"].iloc[0] == 300.0


def test_rfm_features(sample_phase2_df):
    df = add_rfm_features(sample_phase2_df)
    assert "revenue_per_month" in df.columns
    assert "avg_call_duration" in df.columns
    assert df["revenue_per_month"].iloc[0] == pytest.approx(8.333, rel=1e-2)


def test_phase2_business_flags(sample_phase2_df):
    df = add_business_flags(sample_phase2_df)
    assert "is_billing_shock" in df.columns
    assert "is_high_drops" in df.columns
    assert "is_high_value_new" in df.columns
    assert df["is_billing_shock"].iloc[1] == 1  # 70 > 120 * 0.5


def test_sanitized_feature_names(sample_phase2_df):
    pipe = create_preprocessing_pipeline()
    pipe.fit(sample_phase2_df)
    names = get_feature_names(pipe)

    # Verify no special characters remain in feature names
    for name in names:
        assert "<" not in name
        assert ">" not in name
        assert "-" not in name
        assert " " not in name
