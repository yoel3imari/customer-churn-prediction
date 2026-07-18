# End-to-End Customer Churn Prediction — Work Plan

## Objective
Build a portfolio-grade, end-to-end ML project for customer churn prediction using the Cell2Cell dataset. Covers the full ML lifecycle: EDA, feature engineering, multi-algorithm comparison with experiment tracking, explainability, API serving, interactive dashboard, and containerized deployment.

---

## Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.12 | Project baseline |
| Dependency mgmt | `uv` | Fast, modern, pyproject.toml native |
| EDA | Jupyter notebook | Industry standard |
| Data/Features | Pandas, NumPy, Scikit-learn pipelines | Proven, modular |
| Modeling | Scikit-learn, XGBoost, LightGBM, CatBoost | Gradient from interpretable to high-perf |
| Hyperparameter tuning | Optuna | Bayesian optimization, MLflow integration |
| Experiment tracking | MLflow (Postgres backend, local artifact store, Model Registry) | Portfolio differentiator; Postgres already in stack for registry concurrency |
| Explainability | SHAP | Industry standard for model interpretability |
| API | FastAPI + Pydantic | Modern, fast, auto-docs |
| Dashboard | Streamlit | Rapid prototyping, what-if widgets, ML ecosystem fit |
| Resampling | imbalanced-learn (SMOTE) | Synthetic oversampling for churn minority class |
| Containerization | Docker Compose | Reproducible multi-service deployment |
| CI/CD | GitHub Actions | Automated testing |

---

## Dataset: Cell2Cell (Duke University)

- **Rows**: ~71,000
- **Features**: ~70 (usage, demographics, equipment, geography, retention history)
- **Target**: `churn` (binary)
- **Pre-split column**: `traintest` (0/1) — will NOT use; re-splitting with stratification
- **Known concern**: `churndep` column — potential data leakage. User to check docs; will discuss in EDA regardless

---

## Deliverables

| # | Deliverable | Description |
|---|---|---|
| 1 | **EDA Notebook** | Structured narrative: overview → univariate → bivariate → multivariate → missing data → quality summary |
| 2 | **Training Pipeline** | Modular `src/` package with MLflow tracking, Optuna tuning, model comparison |
| 3 | **Model Comparison Report** | Logged in MLflow: 5 algorithms × 2 imbalance strategies + tuned best model |
| 4 | **FastAPI Inference API** | `/predict`, `/predict/batch`, `/model/info`, `/health` |
| 5 | **Streamlit Dashboard** | Data overview, model performance, what-if simulator, batch upload |
| 6 | **Docker Compose** | Postgres + MLflow Server + API + Dashboard |
| 7 | **CI Pipeline** | GitHub Actions: lint + test on push |

---

## Architecture

```
data/raw/cell2cell.csv
        │
        ▼
┌──────────────────────┐
│  01-eda.ipynb         │
│  (exploratory report) │
└──────────────────────┘
        │ (insights → feature decisions)
        ▼
┌────────────────────────────────────────────────────┐
│              Training Pipeline (src/)               │
│                                                     │
│  data/load.py       → load raw CSV                  │
│  data/clean.py      → handle NAs, fix types         │
│  features/build.py  → engineering (interactions,    │
│                        ratios, bins, encodings)     │
│  features/pipeline.py→ sklearn ColumnTransformer    │
│  models/train.py    → train + MLflow autolog        │
│  models/tune.py     → Optuna study                  │
│  models/evaluate.py → metrics, SHAP, plots          │
│  models/compare.py  → compare runs, promote best    │
└──────────────────────┬──────────────────────────────┘
                       │
                        ▼ MLflow Tracking
               ┌─────────────────────────────────┐
               │  MLflow Server (Postgres)        │
               │  :5005                           │
               │                                  │
               │  experiments/ (artifact store)   │
               │  ├─ LogisticRegression           │
               │  │   ├─ Run A: class_weight      │
               │  │   └─ Run B: SMOTE             │
               │  ├─ RandomForest                 │
               │  ├─ XGBoost                      │
               │  ├─ LightGBM                     │
               │  └─ CatBoost                     │
               │  (each x2, then tuned best)      │
               │                                  │
               │  Model Registry:                 │
               │  ChurnPredictor                  │
               │  ├─ Staging (candidate)          │
               │  └─ Production (promoted best)   │
               └───────────┬─────────────────────┘
                          │ loads model by stage
                          ▼
              ┌──────────────────────┐     ┌──────────────────────┐
              │  FastAPI             │◄───►│  Streamlit Dashboard │
              │  :8000               │     │  :8501               │
              │  /predict            │     │  - Overview          │
              │  /predict/batch      │     │  - Model Performance │
              │  /model/info         │     │  - What-if Simulator │
              │  /health             │     │  - Batch Upload      │
              └──────────────────────┘     └──────────────────────┘
                       │                            │
                        │                            │
                        └────────── Docker Compose ──┘
                                   ┌──────────┐
                                   │ Postgres │
                                   │ :5435    │
                                   └────┬─────┘
                                        │
                                   ┌────▼─────┐
                                   │ MLflow   │
                                   │ Server   │
                                   │ :5005    │
                                   └──────────┘
```

---

## EDA Notebook Structure

1. **Dataset Overview**
   - Shape, dtypes, missing %, summary stats
   - Target distribution visualization

2. **Univariate Analysis**
   - Distributions of key features (usage, tenure, demographics)
   - Outlier detection (IQR, domain-based)

3. **Bivariate Analysis (Feature vs Churn)**
   - Numerical: boxplots / KDE by churn status
   - Categorical: churn rate bar charts
   - Statistical tests: chi-squared, t-test, Mann-Whitney U

4. **Multivariate Analysis**
   - Correlation matrix + heatmap
   - Pairplot of top features
   - Interaction exploration (e.g., high revenue + low minutes)

5. **Missing Data Analysis**
   - Missing pattern visualization (missingno)
   - MCAR/MAR/MNAR reasoning
   - Imputation strategy decisions

6. **Data Quality Summary**
   - Anomalies found and resolutions
   - Feature engineering decisions documented
   - Actionable next steps

---

## Feature Engineering Plan

| Category | Features to Engineer |
|---|---|
| **Ratios** | `revenue_per_minute`, `outcall_ratio`, `drop_call_rate` |
| **Aggregations** | `avg_call_duration`, `support_interaction_rate` |
| **Binning** | `tenure_group`, `age_group`, `income_bracket` |
| **Interactions** | `high_value_low_tenure`, `frequent_calls_high_drops` |
| **Encoding** | Target encoding for high-cardinality categoricals, OHE for low-cardinality |
| **Business flags** | `is_high_value` (top decile revenue), `is_at_risk` (multiple flags combined) |

---

## Modeling Strategy

### Algorithms (in order)
1. **LogisticRegression** — baseline, interpretable
2. **RandomForest** — robust ensemble
3. **XGBoost** — industry standard
4. **LightGBM** — fast, handles mixed data
5. **CatBoost** — native categorical support

### Imbalance Handling (each algorithm run twice)
- **Run A — class_weight='balanced'**: Adjusts the loss function so minority misclassifications are penalized ~2.5×. Uses real data only. Supported natively by LogisticRegression, RandomForest (and via scale_pos_weight in XGBoost, is_unbalance in LightGBM).
- **Run B — SMOTE**: Creates synthetic churn examples via k-NN interpolation to produce a balanced training set. Applied as a pipeline step before the classifier. Requires `imbalanced-learn`.

### Optuna Tuning
- Objective: maximize AUC-PR (better for imbalanced data than ROC-AUC)
- Study: 100 trials per algorithm with pruning
- Logged: hyperparameter parallel coordinate plots in MLflow

### Evaluation Metrics
- ROC-AUC, PR-AUC, F1, Precision, Recall
- Confusion matrix at optimal threshold
- Cost-based threshold tuning (FN cost >> FP cost)

### MLflow Tracking & Model Registry

- **Backend**: Postgres (`MLFLOW_TRACKING_URI=postgresql://mlflow:mlflow@postgres:5435/mlflow`)
- **Artifact store**: Local `experiments/` directory (mounted in container)
- **Registry**: Models registered as `ChurnPredictor` after training, promoted to Production via MLflow UI or API after evaluation
- **Services** (Docker Compose):
  - `postgres:16` — dedicated MLflow database
  - `mlflow:2.x` — tracking server + registry UI on port 5005

### Explainability
- SHAP summary plot, force plot, dependence plot
- Feature importance (permutation + built-in)
- LIME on individual predictions (dashboard what-if)

---

## FastAPI Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata, feature names, version, training date |
| POST | `/predict` | Single prediction → churn probability + class + SHAP values |
| POST | `/predict/batch` | CSV upload → predictions file download |

---

## Streamlit Dashboard Pages

1. **Data Overview** — target dist, missing values, feature summaries
2. **Feature Analysis** — interactive EDA plots
3. **Model Performance** — ROC, PR curve, feature importance, confusion matrix, MLflow run comparison table
4. **What-if Simulator** — sliders for key features → live prediction + SHAP waterfall
5. **Batch Predict** — upload CSV → download results

---

## Project Structure (Final)

```
customer-churn-prediction/
├── .github/workflows/   # CI pipeline
├── data/
│   ├── raw/             # Cell2Cell CSV
│   └── processed/       # cleaned/engineered datasets
├── docs/
│   └── work-plan.md     # This file
├── notebooks/
│   └── 01-eda.ipynb     # EDA notebook
├── reports/             # Figures, SHAP plots, PDFs
├── models/              # Serialized .pkl/.joblib files
├── src/
│   ├── __init__.py
│   ├── config.py        # Paths, params, constants
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py      # Data loading
│   │   └── clean.py     # Cleaning, NA handling
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build.py     # Feature engineering functions
│   │   └── pipeline.py  # sklearn ColumnTransformer
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py     # Train + MLflow
│   │   ├── tune.py      # Optuna study
│   │   ├── evaluate.py  # Metrics, SHAP, plots
│   │   └── predict.py   # Inference wrapper
│   └── api/
│       ├── __init__.py
│       ├── app.py       # FastAPI app
│       ├── schemas.py   # Pydantic models
│       └── router.py    # Endpoints
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── docker-compose.yml
├── dashboard/
│   └── app.py           # Streamlit app
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── test_clean.py
│   ├── test_load.py
│   ├── test_build.py
│   ├── test_pipeline.py
│   ├── test_models.py       # TODO: Phase 3
│   └── test_api.py          # TODO: Phase 4
├── pyproject.toml
├── .python-version
├── .gitignore
├── README.md
└── start.sh
```

---

## Execution Order (Phases)

| Phase | What | Depends On |
|---|---|---|
| **0. Setup** | Init `uv` project, install deps, fill `pyproject.toml`, init MLflow tracking dir | Nothing |
| **1. EDA** | Build `01-eda.ipynb` with full narrative | Phase 0 |
| **2. Data + Features** | `src/data/`, `src/features/` packages | Phase 1 (EDA insights) |
| **3a. Prerequisites** | Source `cell2cell.csv`, add `imbalanced-learn`, configure Postgres + MLflow server | Phase 2 |
| **3b. Training Pipeline** | `src/models/`: train, tune, evaluate, compare, register in MLflow | 3a |
| **4. API** | FastAPI app serving registered model from MLflow Model Registry | Phase 3 |
| **5. Dashboard** | Streamlit app consuming API | Phase 4 |
| **6. Docker** | Dockerfiles + docker-compose for Postgres + MLflow + API + Dashboard | Phase 4, 5 |
| **7. CI** | GitHub Actions workflow | Phase 0-6 complete |
| **8. README + Polish** | Final documentation, cleanup | All phases |

---

## Decisions (Confirmed)

- [x] **Imbalance handling**: Run A = `class_weight='balanced'` (or equivalent per model), Run B = SMOTE
- [x] **Model Registry**: MLflow Model Registry with Postgres backend
- [x] **Postgres**: Dedicated `postgres:16` service in Docker Compose for MLflow backend
- [x] **MLflow server**: Runs as a Compose service alongside API and Dashboard

## Open Items

- [x] **Source dataset**: `data/raw/cell2cell.csv` present (19.7 MB, gitignored)
- [ ] **Check `churndep`**: Confirm leakage risk with column docs
- [ ] **Optuna visualizations**: Include MLflow optuna integration plots in docker setup?
- [ ] **Postgres credentials**: Set `MLFLOW_TRACKING_URI` — default `postgresql://mlflow:mlflow@postgres:5435/mlflow` OK?

---
*Updated: 2026-07-15*
