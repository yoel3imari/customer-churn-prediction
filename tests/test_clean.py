import numpy as np
import pandas as pd

from src.config import Columns
from src.data.clean import _median_impute, clean_data


class TestMedianImpute:
    def test_fills_nan_with_median(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        result = _median_impute(df, ["a"])
        assert result["a"].isna().sum() == 0
        assert result["a"].iloc[1] == 2.0

    def test_leaves_other_columns_untouched(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, np.nan, 6.0]})
        result = _median_impute(df, ["a"])
        assert result["b"].isna().sum() == 1

    def test_handles_missing_columns_gracefully(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        result = _median_impute(df, ["nonexistent"])
        assert "nonexistent" not in result.columns

    def test_no_nop_when_no_nans(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = _median_impute(df, ["a"])
        assert result["a"].tolist() == [1.0, 2.0, 3.0]

    def test_empty_cols_list(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        result = _median_impute(df, [])
        assert result["a"].isna().sum() == 1

    def test_multiple_columns(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 3.0]})
        result = _median_impute(df, ["a", "b"])
        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 0

    def test_even_length_column_uses_lower_median(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        result = _median_impute(df, ["a"])
        assert result["a"].iloc[0] == 1.0


class TestCleanData:
    def test_drops_churndep(self, sample_df):
        result = clean_data(sample_df)
        assert "churndep" not in result.columns

    def test_imputes_nans(self, sample_df):
        result = clean_data(sample_df)
        for col in Columns.ALL_MEDIAN_IMPUTE:
            if col in result.columns:
                assert result[col].isna().sum() == 0, f"{col} still has NaN"

    def test_binary_columns_are_int(self, sample_df):
        result = clean_data(sample_df)
        for col in Columns.BINARY_FEATURES:
            if col in result.columns:
                assert result[col].dtype == int, f"{col} is not int"

    def test_returns_copy(self, sample_df):
        original_churndep = sample_df["churndep"].copy()
        clean_data(sample_df)
        assert "churndep" in sample_df.columns
        pd.testing.assert_series_equal(sample_df["churndep"], original_churndep)

    def test_preserves_non_leakage_non_imputed_columns(self, sample_df):
        result = clean_data(sample_df)
        assert "churn" in result.columns
        assert "dropvce" in result.columns
        assert "peakvce" in result.columns

    def test_no_leakage_column_to_drop(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "churn": [0, 1]})
        result = clean_data(df)
        assert list(result.columns) == ["a", "churn"]

    def test_no_nans_present(self):
        df = pd.DataFrame({"revenue": [1.0, 2.0, 3.0], "churn": [0, 1, 0]})
        result = clean_data(df)
        assert result["revenue"].tolist() == [1.0, 2.0, 3.0]

    def test_non_config_columns_not_imputed(self):
        df = pd.DataFrame({"custom_col": [np.nan, 1.0], "churn": [0, 1]})
        result = clean_data(df)
        assert pd.isna(result["custom_col"].iloc[0])
