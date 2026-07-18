import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.data.load import load_data, split_data


@pytest.fixture
def split_df():
    rng = np.random.RandomState(42)
    n = 200
    churn = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    return pd.DataFrame({
        "churn": churn,
        "revenue": rng.uniform(50, 300, n),
        "mou": rng.uniform(100, 1000, n),
        "months": rng.randint(1, 60, n),
    })


class TestSplitData:
    def test_returns_four_elements(self, split_df):
        result = split_data(split_df)
        assert len(result) == 4

    def test_returns_correct_types(self, split_df):
        X_train, X_test, y_train, y_test = split_data(split_df)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_test_size_ratio(self, split_df):
        X_train, X_test, y_train, y_test = split_data(split_df)
        total = len(X_train) + len(X_test)
        ratio = len(X_test) / total
        assert ratio == pytest.approx(0.25, abs=0.05)

    def test_target_not_in_features(self, split_df):
        X_train, X_test, y_train, y_test = split_data(split_df)
        assert "churn" not in X_train.columns
        assert "churn" not in X_test.columns

    def test_preserves_churn_ratio(self, split_df):
        original_ratio = split_df["churn"].mean()
        X_train, X_test, y_train, y_test = split_data(split_df)
        assert y_train.mean() == pytest.approx(original_ratio, abs=0.05)
        assert y_test.mean() == pytest.approx(original_ratio, abs=0.05)

    def test_all_rows_preserved(self, split_df):
        X_train, X_test, y_train, y_test = split_data(split_df)
        assert len(X_train) + len(X_test) == len(split_df)
        assert len(y_train) + len(y_test) == len(split_df)


class TestLoadData:
    def test_drops_specified_columns(self):
        df = pd.DataFrame({
            "Unnamed: 0": [1, 2],
            "X": [10, 20],
            "customer": ["a", "b"],
            "churn": [0, 1],
            "revenue": [100.0, 200.0],
        })
        with patch("src.data.load.pd.read_csv", return_value=df):
            result = load_data()
        assert "Unnamed: 0" not in result.columns
        assert "X" not in result.columns
        assert "customer" not in result.columns

    def test_preserves_informative_columns(self):
        df = pd.DataFrame({
            "Unnamed: 0": [1, 2],
            "churn": [0, 1],
            "revenue": [100.0, 200.0],
        })
        with patch("src.data.load.pd.read_csv", return_value=df):
            result = load_data()
        assert "churn" in result.columns
        assert "revenue" in result.columns

    def test_returns_dataframe(self):
        df = pd.DataFrame({"churn": [0, 1], "revenue": [100.0, 200.0]})
        with patch("src.data.load.pd.read_csv", return_value=df):
            result = load_data()
        assert isinstance(result, pd.DataFrame)

    def test_reads_from_correct_path(self):
        df = pd.DataFrame({"churn": [0, 1]})
        with patch("src.data.load.pd.read_csv", return_value=df) as mock_read:
            load_data()
        mock_read.assert_called_once()

    def test_with_temp_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "Unnamed: 0,customer,revenue,churn\n1,a,100.0,0\n2,b,200.0,1\n"
        )
        with patch("src.data.load.Paths") as mock_paths:
            mock_paths.DATA_RAW = csv_file
            result = load_data()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "Unnamed: 0" not in result.columns
        assert "customer" not in result.columns
