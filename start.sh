#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE_DIR="$SCRIPT_DIR"
COMPOSE="docker compose -f $COMPOSE_DIR/docker-compose.yml"
COMPOSE_MANUAL="$COMPOSE --profile manual"

# Load .env if present (used by docker run below)
if [ -f "$COMPOSE_DIR/.env" ]; then
    set -a; source "$COMPOSE_DIR/.env"; set +a
fi

# Resolve postgres password (default: mlflow)
MLFLOW_PG_PASSWORD="${MLFLOW_PG_PASSWORD:-mlflow}"

# Docker run is the only reliable way to run one-off containers on Colima
# (docker compose run hangs — regression in Docker Compose v2.36 + Colima)
APP_IMAGE="churn-prediction-app:latest"
APP_NETWORK="churn-prediction_default"
APP_PG_HOST="churn-prediction-postgres-1"
APP_MLFLOW_HOST="churn-prediction-mlflow-1"
APP_VOLUME_MLFLOW="churn-prediction_mlflow_artifacts"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
err()   { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    cat <<EOF
Usage: ./start.sh <command> [options]

Commands:
  infra              Start Postgres + MLflow via Docker Compose
  build              Build the app Docker image (after dep changes)
  train              Run full training pipeline (all algorithms x strategies)
  tune <algo>        Tune hyperparameters for a specific algorithm
                     algos: LogisticRegression, RandomForest, XGBoost
  shell              Open Python shell with project imports loaded
  pytest [args]      Run tests (passes args through to pytest)
  jupyter            Start Jupyter notebook server
  dashboard          Start Streamlit dashboard server
  prometheus         Start Prometheus UI server
  help               Show this message

Examples:
  ./start.sh infra                    # start infrastructure
  ./start.sh train                    # full training + compare + promote
  ./start.sh tune XGBoost             # tune XGBoost with Optuna
  ./start.sh pytest tests/ -v         # run tests verbosely
  ./start.sh jupyter                  # launch Jupyter
  ./start.sh dashboard                # launch Streamlit dashboard
  ./start.sh prometheus               # launch Prometheus UI
EOF
    exit 0
}

require_infra() {
    if ! $COMPOSE ps --services --filter "status=running" 2>/dev/null | grep -q "postgres"; then
        warn "Infrastructure not running. Starting it now..."
        cmd_infra
    fi
}

run_app() {
    require_infra
    # Mount only I/O directories, not the full project (Colima bind mount
    # hangs on large trees like .venv via gRPC FUSE).
    docker run --rm \
        -v "$SCRIPT_DIR/src:/app/src" \
        -v "$SCRIPT_DIR/data:/app/data" \
        -v "$SCRIPT_DIR/models:/app/models" \
        -v "$SCRIPT_DIR/reports:/app/reports" \
        -v "$APP_VOLUME_MLFLOW:/mlflow/artifacts" \
        -e MLFLOW_TRACKING_URI="http://${APP_MLFLOW_HOST}:5005" \
        -e PYTHONPATH=/app \
        -e GIT_PYTHON_REFRESH=quiet \
        --network "$APP_NETWORK" \
        "$APP_IMAGE" \
        "$@"
}

cmd_infra() {
    info "Starting Postgres + MLflow..."
    $COMPOSE up -d postgres mlflow
    info "Waiting for Postgres to be healthy..."
    max_retries=30; count=0
    until $COMPOSE exec postgres pg_isready -U mlflow &>/dev/null; do
        sleep 2
        count=$((count + 1))
        if [ $count -ge $max_retries ]; then
            err "Postgres failed to become healthy within $((max_retries * 2)) seconds"
            $COMPOSE logs postgres 2>/dev/null | tail -20
            exit 1
        fi
    done
    ok "Postgres is ready on port 5432"
    ok "MLflow UI at http://localhost:5005"
}

cmd_build() {
    info "Building app image (via Compose)..."
    $COMPOSE_MANUAL build app
    ok "Build complete"
}

cmd_train() {
    info "Running full training pipeline..."
    run_app python3 -c "
from src.data.load import load_data
from src.models.compare import run_full_comparison
df = load_data()
result = run_full_comparison(df)
print('Best algorithm:', result['best_algorithm'])
print('Best strategy:', result['best_strategy'])
print('PR-AUC:', result['final_metrics'].get('pr_auc', 'N/A'))
print('Promoted version:', result['promoted_version'])
"
    ok "Training complete"
}

cmd_tune() {
    local algo="${1:-XGBoost}"
    info "Tuning $algo..."
    run_app python3 -c "
from src.data.load import load_data, split_data
from src.config import ALGORITHMS
from src.models.tune import tune_model

algo_cfg = next(a for a in ALGORITHMS if a['name'] == '$algo')
df = load_data()
X_train, X_test, y_train, y_test = split_data(df)
result = tune_model(X_train, y_train, X_test, y_test, algorithm=algo_cfg['name'], strategy='class_weight', n_trials=100)
print('Best PR-AUC:', result['best_value'])
print('Best params:', result['best_params'])
"
    ok "Tuning complete"
}

cmd_shell() {
    run_app python3 -c "
import os
os.chdir('/app')
import pandas as pd
from src.config import *
from src.data.load import load_data, split_data
from src.data.clean import clean_data
from src.features.build import build_features
from src.features.pipeline import create_preprocessing_pipeline

df = load_data()
X_train, X_test, y_train, y_test = split_data(df)
print(f'Data loaded: {len(df)} rows, {len(df.columns)} columns')
print(f'Train: {len(X_train)}, Test: {len(X_test)}')
print()
import code
code.interact(local=locals())
"
}

cmd_pytest() {
    info "Running tests in Docker container..."
    docker run --rm \
        -v "$SCRIPT_DIR/src:/app/src" \
        -v "$SCRIPT_DIR/tests:/app/tests" \
        -v "$SCRIPT_DIR/data:/app/data" \
        -v "$SCRIPT_DIR/models:/app/models" \
        -v "$SCRIPT_DIR/reports:/app/reports" \
        -v "$APP_VOLUME_MLFLOW:/mlflow/artifacts" \
        -e MLFLOW_TRACKING_URI="http://${APP_MLFLOW_HOST}:5005" \
        -e PYTHONPATH=/app \
        --network "$APP_NETWORK" \
        "$APP_IMAGE" \
        /app/.venv/bin/pytest "$@"
}

cmd_jupyter() {
    require_infra
    info "Starting Jupyter on port 8888..."
    docker run --rm -p 8888:8888 \
        -v "$SCRIPT_DIR/data:/app/data" \
        -v "$SCRIPT_DIR/models:/app/models" \
        -v "$SCRIPT_DIR/reports:/app/reports" \
        -v "$SCRIPT_DIR/notebooks:/app/notebooks" \
        -v "$APP_VOLUME_MLFLOW:/mlflow/artifacts" \
        -e MLFLOW_TRACKING_URI="http://${APP_MLFLOW_HOST}:5005" \
        -e PYTHONPATH=/app \
        --network "$APP_NETWORK" \
        "$APP_IMAGE" \
        jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
}

cmd_dashboard() {
    require_infra
    info "Starting Streamlit Dashboard on port 8501..."
    docker run --rm -p 8501:8501 \
        -v "$SCRIPT_DIR/dashboard:/app/dashboard" \
        -v "$SCRIPT_DIR/reports:/app/reports" \
        -e API_BASE_URL="${API_BASE_URL:-http://localhost:8000/api/v1}" \
        -e MLFLOW_TRACKING_URI="http://${APP_MLFLOW_HOST}:5005" \
        -e PYTHONPATH=/app \

        --network "$APP_NETWORK" \
        "$APP_IMAGE" \
        streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
}

cmd_prometheus() {
    require_infra
    info "Starting Prometheus UI on port 9090..."
    $COMPOSE_MANUAL up -d prometheus
    ok "Prometheus UI available at http://localhost:9090"
}

case "${1:-help}" in
    infra)      shift 0; cmd_infra ;;
    build)      shift 0; cmd_build ;;
    train)      shift 0; cmd_train ;;
    tune)       shift 1; cmd_tune "$@" ;;
    shell)      shift 0; cmd_shell ;;
    pytest)     shift 1; cmd_pytest "$@" ;;
    jupyter)    shift 0; cmd_jupyter ;;
    dashboard)  shift 0; cmd_dashboard ;;
    prometheus) shift 0; cmd_prometheus ;;
    help|--help|-h) usage ;;
    *)          err "Unknown command: $1"; echo; usage ;;
esac

