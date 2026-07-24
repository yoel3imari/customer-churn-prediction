# Customer Churn Prediction

End-to-end machine learning project for predicting customer churn using the **Cell2Cell** dataset (Duke University). Covers the full ML lifecycle: EDA → feature engineering → multi-algorithm comparison → hyperparameter tuning → explainability → API serving → interactive dashboard → containerized deployment.

## Tech Stack

| Layer | Choice | Purpose |
|---|---|---|
| **Language** | Python 3.12 | Project baseline |
| **Dependency management** | `uv` | Fast, modern, `pyproject.toml` native |
| **EDA** | Jupyter Notebook | Structured exploratory analysis |
| **Data / Features** | Pandas, NumPy, scikit-learn | Data wrangling, preprocessing pipelines |
| **Modeling** | scikit-learn, XGBoost, LightGBM, CatBoost | Gradient from interpretable to high-perf |
| **Hyperparameter tuning** | Optuna | Bayesian optimization with MLflow integration |
| **Experiment tracking** | MLflow (Postgres backend) | Full experiment history, model registry |
| **Explainability** | SHAP | Model interpretability (summary, force, waterfall) |
| **API** | FastAPI + Pydantic v2 | Real-time inference with auto-docs |
| **Dashboard** | Streamlit | Interactive what-if simulator, batch upload |
| **Imbalance handling** | imbalanced-learn (SMOTE) | Synthetic oversampling + class weighting |
| **Observability** | Prometheus, `structlog` | Real-time telemetry (/metrics), structured JSON logs |
| **Data Drift** | SciPy (KS-test, $\chi^2$) | Feature data drift detection engine (/monitoring/drift) |
| **Containerization** | Docker Compose | Reproducible multi-service deployment (Postgres + MLflow + API + Dashboard) |
| **CI/CD** | GitHub Actions | Automated testing, Docker build checks, and image publishing |

## Project Structure

```
customer-churn-prediction/
├── .github/workflows/     # CI + CD pipelines
├── data/
│   ├── raw/               # Cell2Cell CSV
│   └── processed/         # Cleaned / engineered datasets
├── dashboard/
│   └── app.py             # Streamlit multi-page dashboard
├── docs/
│   ├── work-plan.md       # Architecture and execution plan
│   ├── analysis.md        # Data dictionary and column guide
│   └── FUTURE_RESEARCH.md # Research notes
├── models/                # Serialized model files
├── notebooks/
│   └── 01-eda.ipynb       # Full EDA with 46 cells, 30+ figures
├── reports/
│   └── figures/           # EDA + evaluation plots
├── src/
│   ├── config.py          # Paths, column definitions, model constants
│   ├── main.py            # CLI entry point
│   ├── data/
│   │   ├── load.py        # CSV loading, stratified split
│   │   └── clean.py       # Median imputation, leakage removal
│   ├── features/
│   │   ├── build.py       # Ratio, binned, and business flag features
│   │   └── pipeline.py    # sklearn ColumnTransformer pipeline
│   ├── models/
│   │   ├── train.py       # MLflow-tracked training
│   │   ├── tune.py        # Optuna hyperparameter optimization
│   │   ├── evaluate.py    # Metrics, plots, SHAP explanations
│   │   ├── compare.py     # Run comparison and model promotion
│   │   └── predict.py     # ChurnPredictor inference wrapper
│   ├── monitoring/
│   │   ├── logging.py     # Structured JSON logging (structlog)
│   │   ├── metrics.py     # Prometheus counters & histograms
│   │   └── drift.py       # Kolmogorov-Smirnov & Chi-Square data drift detector
│   └── api/
│       ├── app.py         # FastAPI application
│       ├── schemas.py     # Pydantic request/response models
│       └── router.py      # API endpoints
├── tests/
│   ├── conftest.py        # Shared test fixtures
│   ├── test_load.py       # Data loading tests
│   ├── test_clean.py      # Data cleaning tests
│   ├── test_build.py      # Feature engineering tests
│   ├── test_pipeline.py   # Pipeline integration tests
│   └── test_monitoring.py # Observability & data drift tests
├── docker-compose.yml     # Multi-service Docker setup
├── Dockerfile             # App container definition
├── Dockerfile.mlflow      # MLflow with Postgres driver
├── pyproject.toml         # Dependencies and project metadata
├── start.sh               # CLI wrapper for common commands
└── README.md              # This file
```

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose (for infrastructure)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd customer-churn-prediction

# Create virtual environment and install dependencies
uv sync

# Start infrastructure (Postgres + MLflow)
./start.sh infra

# Verify: MLflow UI at http://localhost:5005
```

### Run Tests

```bash
# Activate the virtual environment first
source .venv/bin/activate  # or: eval $(uv venv)

# Run all tests
uv run pytest tests/ -v
```

## Usage

### Training Pipeline

Run the full training pipeline (5 algorithms × 2 imbalance strategies, compare, tune best, promote to Production):

```bash
./start.sh train
```

Or use the CLI directly:

```bash
uv run python -m src.main train
```

### Hyperparameter Tuning

```bash
./start.sh tune XGBoost
```

```bash
uv run python -m src.main tune XGBoost --n-trials 100
```

### Start the API

```bash
uv run python -m src.main serve --port 8000
```

API docs available at http://localhost:8000/docs.

### Start the Dashboard

```bash
uv run python -m src.main dashboard --port 8501
```

### Predict from CSV

```bash
uv run python -m src.main predict input.csv -o predictions.csv
```

### Compare MLflow Runs

```bash
uv run python -m src.main compare
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/model/info` | Model metadata, metrics, features |
| GET | `/metrics` | Prometheus metrics telemetry |
| POST | `/api/v1/predict` | Single prediction (optional SHAP) |
| POST | `/api/v1/predict/batch` | CSV upload → batch predictions |
| POST | `/api/v1/monitoring/drift` | CSV feature data drift report |

### Example: Single Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {"revenue": 100, "mou": 500, "months": 12, "custcare": 1},
    "explain": true
  }'
```

## Feature Engineering

| Category | Features |
|---|---|
| **Ratios** | `revenue_per_mou`, `outcall_ratio`, `drop_call_rate`, `support_contact_rate` |
| **Binning** | `tenure_group` (0-6mo, 6-12mo, 1-2yr, 2-4yr, 4-6yr, 6yr+), `age_group` |
| **Business flags** | `is_high_value` (top 10% revenue), `is_low_tenure_high_value`, `is_frequent_drops` |

## Key Business & EDA Insights

Exploratory Data Analysis ([01-eda.ipynb](file:///Users/yusef/projects/portfolio/customer-churn-prediction/notebooks/01-eda.ipynb)) on the ~71,000 Cell2Cell dataset answered these core business questions:

| Business Question | EDA Finding | Key Signal / Feature |
| :--- | :--- | :--- |
| **Which age groups churn the most?** | Younger segments (`<25` and `25–34`) exhibit the highest churn rates; `55+` accounts show higher retention. | `age1` (Age groups) |
| **Does tenure reduce churn?** | **Yes.** Tenure is the #1 numerical predictor. Highest churn risk occurs during the first 0–12 months of service. | `months` (Tenure months) |
| **Do dropped calls increase churn?** | **Yes.** Call quality failures strongly drive churn due to network quality frustration. | `dropvce`, `drop_call_rate` |
| **Does monthly revenue correlate with retention?** | **Yes**, but unexpected overage charges create "bill shock" churn despite high recurring spend. | `revenue`, `overage` |
| **Which service areas churn most?** | Urban (`prizmub`) and Rural (`prizmrur`) accounts churn at higher rates than Suburban (`prizmtwn`) areas. | `prizm*` geography flags |
| **Does customer support usage predict churn?** | **Yes.** Retention and support calls are top 3 churn signals—high contact intensity indicates discontent or save attempts. | `retcalls`, `custcare` |
| **Do usage drops indicate churn?** | **Yes.** Sharp declines in monthly minutes (`changem`) or revenue (`changer`) strongly precede customer departure. | `changem`, `changer` |


## Modeling Strategy

### Algorithms
1. LogisticRegression — interpretable baseline
2. RandomForest — robust ensemble
3. XGBoost — industry standard
4. LightGBM — fast, handles mixed data
5. CatBoost — native categorical support

### Imbalance Handling (each algorithm × 2)
- **Run A**: `class_weight='balanced'` (or equivalent)
- **Run B**: SMOTE synthetic oversampling

### Evaluation
- ROC-AUC, PR-AUC, F1, Precision, Recall
- Confusion matrix at optimal threshold
- SHAP explanations (summary, force, waterfall plots)

## Architecture

```
┌──────────────┐     ┌──────────────────────────────────────┐
│  EDA Notebook│────▶│  Training Pipeline (src/)            │
│  (01-eda)     │     │  load → clean → feature build →     │
└──────────────┘     │  train → tune → evaluate → compare  │
                     └──────────────┬───────────────────────┘
                                    │ MLflow tracking
                                    ▼
                     ┌──────────────────────────────────────┐
                     │  MLflow Server (Postgres, :5005)     │
                     │  Model Registry: ChurnPredictor      │
                     └──────────────┬───────────────────────┘
                                    │ loads Production model
                                    ▼
               ┌──────────────────────┐    ┌──────────────────┐
               │  FastAPI (:8000)     │◄──►│  Streamlit (:8501)│
               │  /predict            │    │  Dashboard       │
               │  /predict/batch      │    └──────────────────┘
               │  /model/info         │
               │  /health             │
               └──────────────────────┘
```

### Docker Services

| Service | Image | Port |
|---------|-------|------|
| Postgres | postgres:16-alpine | 5435 |
| MLflow | Custom (with psycopg2) | 5005 |
| App | Custom (via Dockerfile) | 8000 |
| Dashboard | Custom (via Streamlit) | 8501 |

## Dataset: Cell2Cell

- **Rows**: ~71,000
- **Features**: ~70 (usage, demographics, equipment, geography, retention history)
- **Target**: `churn` (binary: 0 = stayed, 1 = churned)
- **Churn rate**: ~29%

## License

MIT

