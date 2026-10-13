"""Statistical Data Drift detection for continuous and categorical features."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from src.config import Columns

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """Statistical data drift detector comparing reference data vs current batch."""

    def __init__(
        self,
        reference_data: pd.DataFrame | None = None,
        alpha: float = 0.05,
    ) -> None:
        """Initialize the drift detector.

        Args:
            reference_data: Baseline training DataFrame. If None, mock/baseline is loaded on first check.
            alpha: Significance threshold for p-value (default 0.05). P-value < alpha indicates drift.
        """
        self.alpha = alpha
        self.reference_data = reference_data

    def set_reference_data(self, reference_data: pd.DataFrame) -> None:
        """Update reference baseline dataset."""
        self.reference_data = reference_data.copy()

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        """Detect feature data drift between reference data and current inference batch.

        Args:
            current_data: DataFrame containing recent prediction request features.
            features: Optional list of features to evaluate. Defaults to Columns.ALL_FEATURES if present.

        Returns:
            Dictionary containing overall drift summary and per-feature drift details.
        """
        if current_data.empty:
            return {
                "dataset_drift_detected": False,
                "drift_share": 0.0,
                "number_of_drifted_features": 0,
                "number_of_features": 0,
                "features": {},
            }

        if self.reference_data is None or self.reference_data.empty:
            # Fallback: create mock baseline from current data if reference isn't provided
            logger.warning("No reference dataset provided for drift detection; using baseline snapshot.")
            self.reference_data = current_data.copy()

        if features is None:
            features = [c for c in Columns.ALL_FEATURES if c in current_data.columns and c in self.reference_data.columns]
            if not features:
                features = [c for c in current_data.columns if c in self.reference_data.columns]

        feature_results: dict[str, dict[str, Any]] = {}
        drifted_count = 0

        for col in features:
            ref_col = self.reference_data[col].dropna()
            curr_col = current_data[col].dropna()

            if len(ref_col) < 5 or len(curr_col) < 5:
                continue

            # Determine whether feature is continuous or categorical/binary
            is_numeric = np.issubdtype(ref_col.dtype, np.number) and ref_col.nunique() > 10

            if is_numeric:
                stat, p_value = ks_2samp(ref_col, curr_col)
                test_name = "kolmogorov_smirnov"
            else:
                # Build contingency table for categorical/binary
                ref_counts = ref_col.value_counts()
                curr_counts = curr_col.value_counts()
                categories = list(set(ref_counts.index).union(set(curr_counts.index)))

                ref_freq = [ref_counts.get(cat, 0) for cat in categories]
                curr_freq = [curr_counts.get(cat, 0) for cat in categories]

                contingency = np.array([ref_freq, curr_freq])
                try:
                    stat, p_value, _, _ = chi2_contingency(contingency)
                    test_name = "chi_square"
                except ValueError:
                    stat, p_value = 0.0, 1.0
                    test_name = "chi_square_failed"

            is_drifted = bool(p_value < self.alpha)
            if is_drifted:
                drifted_count += 1

            feature_results[col] = {
                "drift_detected": is_drifted,
                "p_value": float(p_value),
                "statistic": float(stat),
                "test_name": test_name,
            }

        total_features = len(feature_results)
        drift_share = round(drifted_count / total_features, 4) if total_features > 0 else 0.0
        overall_drift = drift_share >= 0.3  # Dataset drifts if >= 30% features drift

        return {
            "dataset_drift_detected": overall_drift,
            "drift_share": drift_share,
            "number_of_drifted_features": drifted_count,
            "number_of_features": total_features,
            "features": feature_results,
        }
