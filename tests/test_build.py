import numpy as np
import pandas as pd

from src.features.build import (
    add_binned_features,
    add_business_flags,
    add_ratio_features,
    build_features,
)


class TestAddRatioFeatures:
    def test_all_ratio_columns_exist(self, sample_df_clean):
        result = add_ratio_features(sample_df_clean)
        expected = ["revenue_per_mou", "drop_call_rate", "outcall_ratio", "support_contact_rate"]
        for col in expected:
            assert col in result.columns

    def test_ratio_columns_are_float(self, sample_df_clean):
        result = add_ratio_features(sample_df_clean)
        for col in ["revenue_per_mou", "drop_call_rate", "outcall_ratio", "support_contact_rate"]:
            assert result[col].dtype == np.float64

    def test_zero_division_produces_nan(self):
        df = pd.DataFrame({
            "revenue": [10.0, 20.0],
            "mou": [0.0, 0.0],
            "dropvce": [0.0, 0.0],
            "peakvce": [0.0, 0.0],
            "opeakvce": [0.0, 0.0],
            "outcalls": [0.0, 0.0],
            "incalls": [0.0, 0.0],
            "custcare": [1.0, 0.0],
            "retcalls": [0.0, 1.0],
            "months": [0.0, 0.0],
        })
        result = add_ratio_features(df)
        assert result["revenue_per_mou"].isna().all()
        assert result["drop_call_rate"].isna().all()
        assert result["outcall_ratio"].isna().all()
        assert result["support_contact_rate"].isna().all()

    def test_does_not_mutate_input(self, sample_df_clean):
        original = sample_df_clean.copy()
        add_ratio_features(sample_df_clean)
        pd.testing.assert_frame_equal(sample_df_clean, original)

    def test_missing_columns_skips_gracefully(self):
        df = pd.DataFrame({"revenue": [100.0], "mou": [50.0]})
        result = add_ratio_features(df)
        assert "revenue_per_mou" in result.columns
        assert "drop_call_rate" not in result.columns
        assert "outcall_ratio" not in result.columns
        assert "support_contact_rate" not in result.columns


class TestAddBinnedFeatures:
    def test_tenure_group_is_categorical(self, sample_df_clean):
        result = add_binned_features(sample_df_clean)
        assert result["tenure_group"].dtype.name == "category"

    def test_age_group_is_categorical(self, sample_df_clean):
        result = add_binned_features(sample_df_clean)
        assert result["age_group"].dtype.name == "category"

    def test_tenure_group_labels(self, sample_df_clean):
        result = add_binned_features(sample_df_clean)
        expected_labels = ["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4-6yr", "6yr+"]
        assert list(result["tenure_group"].cat.categories) == expected_labels

    def test_age_group_labels(self, sample_df_clean):
        result = add_binned_features(sample_df_clean)
        expected_labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
        assert list(result["age_group"].cat.categories) == expected_labels

    def test_tenure_bin_edges(self):
        df = pd.DataFrame({
            "months": [1, 3, 6, 7, 12, 13, 24, 25, 48, 49, 72, 100],
            "age1": [30] * 12,
        })
        result = add_binned_features(df)
        assert result["tenure_group"].tolist() == [
            "0-6mo", "0-6mo", "0-6mo",
            "6-12mo", "6-12mo",
            "1-2yr", "1-2yr",
            "2-4yr", "2-4yr",
            "4-6yr", "4-6yr",
            "6yr+",
        ]

    def test_age_bin_edges(self):
        df = pd.DataFrame({
            "age1": [10, 26, 36, 46, 56, 66, 80],
            "months": [12] * 7,
        })
        result = add_binned_features(df)
        assert result["age_group"].tolist() == [
            "<25", "25-34", "35-44", "45-54", "55-64", "65+", "65+",
        ]

    def test_missing_months_skips_tenure(self):
        df = pd.DataFrame({"age1": [30, 40]})
        result = add_binned_features(df)
        assert "tenure_group" not in result.columns
        assert "age_group" in result.columns

    def test_missing_age1_skips_age(self):
        df = pd.DataFrame({"months": [12, 24]})
        result = add_binned_features(df)
        assert "age_group" not in result.columns
        assert "tenure_group" in result.columns


class TestAddBusinessFlags:
    def test_flags_are_0_or_1_int(self, sample_df_clean):
        df = add_ratio_features(sample_df_clean)
        result = add_business_flags(df)
        for col in ["is_high_value", "is_low_tenure_high_value", "is_frequent_drops"]:
            assert col in result.columns
            assert result[col].dtype in (np.int64, int)
            assert set(result[col].unique()).issubset({0, 1})

    def test_is_high_value_flags_top_10_percent(self):
        df = pd.DataFrame({
            "revenue": list(range(1, 101)),
            "months": [12] * 100,
            "drop_call_rate": [0.0] * 100,
        })
        result = add_business_flags(df)
        assert result["is_high_value"].sum() == 10
        assert result.loc[result["revenue"] >= 91, "is_high_value"].eq(1).all()
        assert result.loc[result["revenue"] <= 90, "is_high_value"].eq(0).all()

    def test_high_value_quantile_boundary(self):
        df = pd.DataFrame({
            "revenue": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "months": [12] * 10,
        })
        result = add_business_flags(df)
        q90 = df["revenue"].quantile(0.9)
        assert result.loc[result["revenue"] >= q90, "is_high_value"].eq(1).all()
        assert result.loc[result["revenue"] < q90, "is_high_value"].eq(0).all()

    def test_low_tenure_high_value(self):
        df = pd.DataFrame({
            "revenue": [100.0, 50.0, 200.0, 30.0],
            "months": [6, 12, 3, 24],
        })
        result = add_business_flags(df)
        expected = [0, 0, 1, 0]
        assert result["is_low_tenure_high_value"].tolist() == expected

    def test_is_frequent_drops(self):
        df = pd.DataFrame({
            "revenue": [50.0] * 10,
            "drop_call_rate": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0],
        })
        result = add_business_flags(df)
        q90 = df["drop_call_rate"].quantile(0.9)
        assert result["is_frequent_drops"].sum() >= 1
        assert result.loc[result["drop_call_rate"] >= q90, "is_frequent_drops"].eq(1).all()

    def test_missing_revenue_skips_flags(self):
        df = pd.DataFrame({"months": [12, 24], "drop_call_rate": [0.1, 0.2]})
        result = add_business_flags(df)
        assert "is_high_value" not in result.columns
        assert "is_low_tenure_high_value" not in result.columns
        assert "is_frequent_drops" in result.columns

    def test_missing_drop_call_rate_skips_frequent_drops(self):
        df = pd.DataFrame({"revenue": [100.0, 200.0], "months": [6, 12]})
        result = add_business_flags(df)
        assert "is_frequent_drops" not in result.columns


class TestBuildFeatures:
    def test_all_expected_columns_present(self, sample_df_clean):
        result = build_features(sample_df_clean)
        expected_cols = [
            "revenue_per_mou", "drop_call_rate", "outcall_ratio", "support_contact_rate",
            "tenure_group", "age_group",
            "is_high_value", "is_low_tenure_high_value", "is_frequent_drops",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_composes_all_feature_sets(self, sample_df_clean):
        result = build_features(sample_df_clean)
        assert result["tenure_group"].dtype.name == "category"
        assert result["revenue_per_mou"].dtype == np.float64
        assert result["is_high_value"].dtype in (np.int64, int)

    def test_does_not_mutate_input(self, sample_df_clean):
        original = sample_df_clean.copy()
        build_features(sample_df_clean)
        pd.testing.assert_frame_equal(sample_df_clean, original)

    def test_original_columns_preserved(self, sample_df_clean):
        result = build_features(sample_df_clean)
        for col in sample_df_clean.columns:
            assert col in result.columns
