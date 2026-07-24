"""scikit-learn preprocessing pipeline for the churn prediction feature set.

Composes data cleaning, feature engineering, scaling, and one-hot encoding
into a single importable ``Pipeline`` ready for model training.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.preprocessing._function_transformer import FunctionTransformer

from src.config import Columns
from src.data.clean import clean_data
from src.features.build import build_features

# Categorical columns produced by ``build_features`` that need one-hot encoding.
BINNED_COLS: list[str] = ["tenure_group", "age_group"]


def create_preprocessing_pipeline() -> Pipeline:
    """Build the full preprocessing pipeline: clean -> features -> scale/OHE."""
    preprocessor = ColumnTransformer(
        sparse_threshold=0,  # dense output — SMOTE and other models need it
        transformers=[
            ("scale", RobustScaler(), Columns.ALL_CONTINUOUS),
            (
                "ohe",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                BINNED_COLS,
            ),
        ],
        remainder="passthrough",
    )

    return Pipeline(
        steps=[
            ("clean", FunctionTransformer(clean_data)),
            ("features", FunctionTransformer(build_features)),
            ("preprocessor", preprocessor),
        ]
    )


import re


def sanitize_feature_name(name: str) -> str:
    """Sanitize feature names by replacing special JSON/regex chars with underscores."""
    return re.sub(r"[^\w]", "_", name)


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Return the output feature names of a fitted preprocessing pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    raw_names = list(preprocessor.get_feature_names_out())
    return [sanitize_feature_name(n) for n in raw_names]
