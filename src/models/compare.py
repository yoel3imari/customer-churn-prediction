"""Compare trained MLflow runs and promote the best model to Production."""

import time

import pandas as pd
from mlflow.client import MlflowClient

import src.config as config
import mlflow


def fetch_experiment_runs(experiment_name: str | None = None) -> list[dict]:
    experiment_name = experiment_name or config.MLFLOW_EXPERIMENT_NAME
    client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
    )
    result: list[dict] = []
    for run in runs:
        result.append(
            {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
                "tags": dict(run.data.tags),
                "status": run.info.status,
            }
        )
    return result


def _build_comparison_df(runs: list[dict]) -> pd.DataFrame:
    valid_algos = {a["name"] for a in config.ALGORITHMS} | {"SoftVotingEnsemble"}
    rows: list[dict] = []
    for r in runs:
        params = r.get("params", {})
        metrics = r.get("metrics", {})
        algo = params.get("model_algorithm") or params.get("algorithm")
        if algo not in valid_algos:
            continue
        rows.append(
            {
                "run_id": r.get("run_id"),
                "algorithm": algo,
                "strategy": params.get("imbalance_strategy") or params.get("strategy"),
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "f1": metrics.get("f1"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "run_name": r.get("run_name"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return df


def compare_runs(experiment_name: str | None = None) -> pd.DataFrame:
    runs = fetch_experiment_runs(experiment_name)
    return _build_comparison_df(runs)


def select_best_run(runs: list[dict] | None = None, metric: str = "pr_auc") -> str:
    if runs is None:
        runs = fetch_experiment_runs()
    df = _build_comparison_df(runs)
    if df.empty:
        raise RuntimeError("No successful runs available to select the best run.")
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in run data.")
    df = df.sort_values(metric, ascending=False)
    return str(df.iloc[0]["run_id"])


def promote_to_production(run_id: str, stage: str = "Production") -> None:
    client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
    model_name = config.MLFLOW_REGISTRY_NAME
    model_uri = f"runs:/{run_id}/model"

    existing = client.search_model_versions(f"run_id = '{run_id}'")
    if existing:
        version = existing[0].version
    else:
        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
        )
        version = mv.version

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )


def run_full_comparison(dataset_df: pd.DataFrame) -> dict:
    from src.data.load import split_data
    from src.models.train import train_model
    from src.models.tune import tune_model

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = split_data(dataset_df)

    total_runs = len(config.ALGORITHMS) * len(config.IMBALANCE_STRATEGIES)
    run_count = 0
    overall_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Starting full comparison: {total_runs} runs")
    print(f"  Algorithms: {', '.join(a['name'] for a in config.ALGORITHMS)}")
    print(f"  Strategies: {', '.join(config.IMBALANCE_STRATEGIES)}")
    print(f"{'='*60}\n")

    for algorithm in config.ALGORITHMS:
        print("=" * 10, algorithm.get("name", "Algorithm"), "=" * 10)
        for strategy in config.IMBALANCE_STRATEGIES:
            run_count += 1
            algo_name = algorithm["name"]
            iter_start = time.time()

            print(
                f"  [{run_count:>2}/{total_runs}] "
                f"Training {algo_name:20s} | {strategy:13s}  "
                f"...  ", end="", flush=True
            )

            try:
                result = train_model(
                    X_train=X_train, y_train=y_train,
                    X_val=X_test, y_val=y_test,
                    algorithm=algo_name, strategy=strategy,
                    run_name=f"{algo_name}_{strategy}",
                )
                metrics = result["metrics"]
                elapsed = time.time() - iter_start
                pr_auc = metrics.get("pr_auc", 0)
                print(f"done  ({elapsed:6.1f}s)  PR-AUC: {pr_auc:.4f}")
            except Exception as exc:
                elapsed = time.time() - iter_start
                print(f"FAILED ({elapsed:6.1f}s)  Error: {exc}")

    # ── Train Soft-Voting Ensemble ──────────────────────────────────────────
    print("=" * 10, "SoftVotingEnsemble", "=" * 10)
    ens_start = time.time()
    print("  Training SoftVotingEnsemble (LR + XGB + LGBM) ...  ", end="", flush=True)
    try:
        from src.models.train import train_soft_voting_ensemble
        ens_res = train_soft_voting_ensemble(X_train, y_train, X_test, y_test)
        ens_elapsed = time.time() - ens_start
        ens_pr_auc = ens_res["metrics"].get("pr_auc", 0)
        print(f"done  ({ens_elapsed:6.1f}s)  PR-AUC: {ens_pr_auc:.4f}")
    except Exception as exc:
        ens_elapsed = time.time() - ens_start
        print(f"FAILED ({ens_elapsed:6.1f}s)  Error: {exc}")

    total_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"  All {total_runs} runs complete in {total_elapsed:.1f}s")
    print(f"{'='*60}\n")

    runs = fetch_experiment_runs()
    comparison_df = _build_comparison_df(runs)
    best_run_id = select_best_run(runs, metric="pr_auc")

    best_row = comparison_df.iloc[0]
    best_algorithm_name = str(best_row["algorithm"])
    best_strategy = str(best_row["strategy"])

    print(f"\n{'='*60}")
    print(f"  Best: {best_algorithm_name} + {best_strategy}")
    print(f"  PR-AUC: {best_row['pr_auc']:.4f}")
    print(f"{'='*60}\n")
    print(f"  Tuning {best_algorithm_name} with Optuna (100 trials)...")
    tune_start = time.time()

    tuned_params = tune_model(X_train, y_train, X_test, y_test,
                              algorithm=best_algorithm_name,
                              strategy=best_strategy)

    tune_elapsed = time.time() - tune_start
    print(f"  Tuning complete in {tune_elapsed:.1f}s")
    print(f"  Best PR-AUC from tuning: {tuned_params['best_value']:.4f}\n")

    print(f"  Training final model with tuned params...")
    final_train_start = time.time()

    final_result = train_model(
        X_train=X_train, y_train=y_train,
        X_val=X_test, y_val=y_test,
        algorithm=best_algorithm_name,
        strategy=best_strategy,
        params=tuned_params["best_params"],
        run_name=f"{best_algorithm_name}_{best_strategy}_final",
    )
    final_metrics = final_result["metrics"]
    final_run_id = final_result["run_id"]

    final_train_elapsed = time.time() - final_train_start
    promote_to_production(final_run_id)

    import joblib
    config.Paths.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_result["model"], config.Paths.MODELS / "model.joblib")
    joblib.dump(final_result["preprocessor"], config.Paths.MODELS / "preprocessing_pipeline.joblib")

    client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
    versions = client.search_model_versions(f"run_id = '{final_run_id}'")
    promoted_version = int(versions[0].version) if versions else None

    total_elapsed_final = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"  ✅ Pipeline complete in {total_elapsed_final:.1f}s")
    print(f"  Best algorithm : {best_algorithm_name}")
    print(f"  Best strategy  : {best_strategy}")
    print(f"  Final PR-AUC   : {final_metrics.get('pr_auc', 'N/A'):.4f}")
    print(f"  Promoted version: {promoted_version}")
    print(f"{'='*60}\n")

    return {
        "comparison_df": comparison_df,
        "best_run_id": best_run_id,
        "best_algorithm": best_algorithm_name,
        "best_strategy": best_strategy,
        "tuned_params": tuned_params,
        "final_metrics": final_metrics,
        "promoted_version": promoted_version,
    }
