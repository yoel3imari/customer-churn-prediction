"""CLI entry point for the customer churn prediction pipeline."""

from __future__ import annotations

import argparse
import sys


def cmd_train(args: argparse.Namespace) -> None:
    """Run full training pipeline: train all algorithms x strategies, compare, tune best, promote."""
    from src.data.load import load_data
    from src.models.compare import run_full_comparison

    df = load_data()
    result = run_full_comparison(df)

    print(f"\n{'='*60}")
    print(f"  Best algorithm : {result['best_algorithm']}")
    print(f"  Best strategy  : {result['best_strategy']}")
    print(f"  Final PR-AUC   : {result['final_metrics'].get('pr_auc', 'N/A'):.4f}")
    print(f"  Promoted version: {result['promoted_version']}")
    print(f"{'='*60}")

    print("\nComparison table:")
    print(result["comparison_df"].to_string(index=False))


def cmd_tune(args: argparse.Namespace) -> None:
    """Tune hyperparameters for a specific algorithm via Optuna."""
    from src.data.load import load_data, split_data
    from src.config import ALGORITHMS
    from src.models.tune import tune_model

    valid_names = {a["name"] for a in ALGORITHMS}
    if args.algorithm not in valid_names:
        print(f"Error: unknown algorithm '{args.algorithm}'")
        print(f"Valid options: {', '.join(sorted(valid_names))}")
        sys.exit(1)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    result = tune_model(
        X_train,
        y_train,
        X_test,
        y_test,
        algorithm=args.algorithm,
        strategy=args.strategy,
        n_trials=args.n_trials,
    )

    print(f"\n{'='*60}")
    print(f"  Algorithm : {args.algorithm}")
    print(f"  Strategy  : {args.strategy}")
    print(f"  Best PR-AUC: {result['best_value']:.4f}")
    print(f"  Best params: {result['best_params']}")
    print(f"{'='*60}")


def cmd_predict(args: argparse.Namespace) -> None:
    """Run batch predictions from a CSV file or stdin."""
    import pandas as pd
    from src.models.predict import ChurnPredictor

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        df = pd.read_csv(sys.stdin)

    predictor = ChurnPredictor(stage=args.stage, model_uri=args.model_uri)
    result = predictor.predict_batch(df)

    output = pd.concat([df, result], axis=1)

    if args.output:
        output.to_csv(args.output, index=False)
        print(f"Predictions written to {args.output}")
    else:
        print(output.to_string())


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI prediction server."""
    import uvicorn

    print(f"Starting API server on port {args.port}...")
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
    )


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Start the Streamlit dashboard."""
    import subprocess

    dashboard_path = str(
        __import__("pathlib").Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    )
    cmd = ["streamlit", "run", dashboard_path, "--server.port", str(args.port)]
    print(f"Starting dashboard on port {args.port}...")
    subprocess.run(cmd, check=True)


def cmd_compare(args: argparse.Namespace) -> None:
    """List MLflow runs sorted by PR-AUC."""
    from src.models.compare import compare_runs

    df = compare_runs()

    if df.empty:
        print("No finished runs found in MLflow.")
        return

    print(f"\nMLflow Runs (sorted by PR-AUC):")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}")
    print(f"Total runs: {len(df)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="churn",
        description="Customer Churn Prediction — CLI entry point",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ────────────────────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Run full training pipeline (all algorithms x strategies)")
    train_p.set_defaults(func=cmd_train)

    # ── tune ─────────────────────────────────────────────────────────────
    tune_p = sub.add_parser("tune", help="Tune hyperparameters for a specific algorithm")
    tune_p.add_argument("algorithm", help="Algorithm name (e.g. XGBoost, LightGBM)")
    tune_p.add_argument(
        "--strategy",
        default="class_weight",
        choices=["class_weight", "smote"],
        help="Imbalance strategy (default: class_weight)",
    )
    tune_p.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials (default: 100)",
    )
    tune_p.set_defaults(func=cmd_tune)

    # ── predict ──────────────────────────────────────────────────────────
    predict_p = sub.add_parser("predict", help="Run batch predictions from CSV or stdin")
    predict_p.add_argument("csv", nargs="?", help="Path to input CSV (reads stdin if omitted)")
    predict_p.add_argument("-o", "--output", help="Path to write output CSV")
    predict_p.add_argument(
        "--stage",
        default="Production",
        help="MLflow model stage (default: Production)",
    )
    predict_p.add_argument("--model-uri", help="Override model URI (e.g. models:/Name/Version)")
    predict_p.set_defaults(func=cmd_predict)

    # ── serve ────────────────────────────────────────────────────────────
    serve_p = sub.add_parser("serve", help="Start the FastAPI prediction server")
    serve_p.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_p.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve_p.set_defaults(func=cmd_serve)

    # ── dashboard ────────────────────────────────────────────────────────
    dash_p = sub.add_parser("dashboard", help="Start the Streamlit dashboard")
    dash_p.add_argument("--port", type=int, default=8501, help="Port (default: 8501)")
    dash_p.set_defaults(func=cmd_dashboard)

    # ── compare ──────────────────────────────────────────────────────────
    comp_p = sub.add_parser("compare", help="List MLflow runs sorted by PR-AUC")
    comp_p.set_defaults(func=cmd_compare)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
