"""Paths, column definitions, and constants for the churn prediction pipeline."""

import os
from pathlib import Path
from typing import ClassVar

import numpy as np


# ── Paths ──────────────────────────────────────────────────────────────────
class Paths:
    """Project-relative paths. All derive from the project root."""

    ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_RAW: Path = ROOT / "data" / "raw" / "cell2cell.csv"
    DATA_PROCESSED: Path = ROOT / "data" / "processed"
    MODELS: Path = ROOT / "models"
    REPORTS_FIGURES: Path = ROOT / "reports" / "figures"
    MLFLOW_DIR: Path = ROOT / "experiments"


# ── Column groups ──────────────────────────────────────────────────────────
class Columns:
    """Named column groups used throughout the pipeline."""

    # Non-informative — dropped on load
    DROP: ClassVar[list[str]] = ["Unnamed: 0", "X", "customer"]

    # Target
    TARGET: str = "churn"

    # Leakage — drop from feature matrix
    LEAKAGE: ClassVar[list[str]] = ["churndep"]

    # Train / test split indicator (kept for reference, not used for splitting)
    SPLIT_FLAG: str = "traintest"

    # ── Features that need median imputation (EDA §5.2) ──
    BILLING_USAGE_MEDIAN_IMPUTE: ClassVar[list[str]] = [
        "revenue",
        "mou",
        "recchrge",
        "directas",
        "overage",
        "roam",
    ]

    CHANGE_MEDIAN_IMPUTE: ClassVar[list[str]] = [
        "changem",
        "changer",
    ]

    DEVICE_MEDIAN_IMPUTE: ClassVar[list[str]] = [
        "phones",
        "models",
        "eqpdays",
    ]

    AGE_MEDIAN_IMPUTE: ClassVar[list[str]] = [
        "age1",
        "age2",
    ]

    ALL_MEDIAN_IMPUTE: ClassVar[list[str]] = (
        BILLING_USAGE_MEDIAN_IMPUTE
        + CHANGE_MEDIAN_IMPUTE
        + DEVICE_MEDIAN_IMPUTE
        + AGE_MEDIAN_IMPUTE
    )

    # ── Continuous features used in histograms / univariate ──
    CONTINUOUS: ClassVar[list[str]] = [
        "revenue",
        "mou",
        "recchrge",
        "directas",
        "overage",
        "roam",
        "changem",
        "changer",
        "mourec",
        "peakvce",
        "opeakvce",
        "outcalls",
        "incalls",
        "dropvce",
        "blckvce",
        "unansvce",
        "dropblk",
        "months",
        "eqpdays",
        "age1",
        "income",
        "setprc",
    ]

    # ── Binary / categorical flags ──
    BINARY_FEATURES: ClassVar[list[str]] = [
        "refurb",
        "webcap",
        "newcelly",
        "newcelln",
        "truck",
        "rv",
        "pcown",
        "ownrent",
        "mcycle",
        "occprof",
        "occcler",
        "occcrft",
        "occstud",
        "occhmkr",
        "occret",
        "occself",
        "marryyes",
        "marryun",
        "prizmrur",
        "prizmub",
        "prizmtwn",
        "mailord",
        "mailres",
        "mailflag",
        "travel",
        "creditcd",
        "retcall",
        "retaccpt",
        "retcalls",
        "credita",
        "creditaa",
        "children",
        "incmiss",
        "setprcm",
    ]

    # ── One-hot blocks (mutually exclusive per row) ──
    OCCUPATION_COLS: ClassVar[list[str]] = [
        "occprof",
        "occcler",
        "occcrft",
        "occstud",
        "occhmkr",
        "occret",
        "occself",
    ]

    GEOGRAPHY_COLS: ClassVar[list[str]] = [
        "prizmrur",
        "prizmub",
        "prizmtwn",
    ]

    MARITAL_COLS: ClassVar[list[str]] = [
        "marryyes",
        "marryun",
    ]

    # ── Retention / high-value signals ──
    RETENTION_COLS: ClassVar[list[str]] = [
        "retcalls",
        "retaccpt",
        "retcall",
    ]

    # ── All feature columns (everything except target + leakage + split) ──
    ALL_FEATURES: ClassVar[list[str]] = [
        "revenue",
        "mou",
        "recchrge",
        "directas",
        "overage",
        "roam",
        "changem",
        "changer",
        "dropvce",
        "blckvce",
        "unansvce",
        "custcare",
        "threeway",
        "mourec",
        "outcalls",
        "incalls",
        "peakvce",
        "opeakvce",
        "dropblk",
        "callfwdv",
        "callwait",
        "months",
        "uniqsubs",
        "actvsubs",
        "phones",
        "models",
        "eqpdays",
        "age1",
        "age2",
        "children",
        "credita",
        "creditaa",
        "creditcd",
        "prizmrur",
        "prizmub",
        "prizmtwn",
        "refurb",
        "webcap",
        "truck",
        "rv",
        "occprof",
        "occcler",
        "occcrft",
        "occstud",
        "occhmkr",
        "occret",
        "occself",
        "ownrent",
        "marryun",
        "marryyes",
        "mailord",
        "mailres",
        "mailflag",
        "travel",
        "pcown",
        "retcalls",
        "retaccpt",
        "newcelly",
        "newcelln",
        "refer",
        "incmiss",
        "income",
        "mcycle",
        "setprcm",
        "setprc",
        "retcall",
    ]


# ── Modeling constants ────────────────────────────────────────────────────
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.25
TARGET_CHURN_RATE: float = 0.29  # ≈ observed in full dataset

# MLflow tracking (overridable via env var)
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5005",
)
MLFLOW_EXPERIMENT_NAME: str = "churn-prediction"
MLFLOW_REGISTRY_NAME: str = "ChurnPredictor"

# Model definitions
ALGORITHMS: list[dict] = [
    {"name": "LogisticRegression",     "class": "sklearn.linear_model.LogisticRegression",     "is_sklearn": True},
    {"name": "RandomForest",           "class": "sklearn.ensemble.RandomForestClassifier",     "is_sklearn": True},
    {"name": "XGBoost",                "class": "xgboost.XGBClassifier",                       "is_sklearn": False},
]

# Imbalance strategies
IMBALANCE_STRATEGIES: list[str] = ["class_weight", "smote"]

# Columns that must be 0/1 integer
BINARY_INT_COLS: list[str] = Columns.BINARY_FEATURES

# Income ordinal mapping
INCOME_MAP: dict[int, str] = {
    0: "Unknown",
    1: "Lowest",
    2: "Low",
    3: "Lower-middle",
    4: "Middle",
    5: "Upper-middle",
    6: "High",
    7: "Highest",
    8: "Highest",
    9: "Highest",
}

# Tenure bin edges (months)
TENURE_BINS: list[int] = [0, 6, 12, 24, 48, 72, np.iinfo(np.int64).max]
TENURE_LABELS: list[str] = [
    "0-6mo",
    "6-12mo",
    "1-2yr",
    "2-4yr",
    "4-6yr",
    "6yr+",
]

# Age bin edges
AGE_BINS: list[int] = [0, 25, 35, 45, 55, 65, np.iinfo(np.int64).max]
AGE_LABELS: list[str] = [
    "<25",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65+",
]
