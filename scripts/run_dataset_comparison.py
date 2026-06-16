from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expense_modeling.clustering import run_clustering
from expense_modeling.config import DEFAULT_NUM_PEOPLE, DEFAULT_SEED, METRICS_DIR, ensure_directories
from expense_modeling.data_generator import generate_transactions
from expense_modeling.data_generator_realistic import generate_realistic_dataset
from expense_modeling.preprocessing import aggregate_daily_transactions
from expense_modeling.supervised import train_supervised_models


DATASET_MATRIX = (
    {"dataset_name": "baseline_synthetic", "generator": "baseline", "scenario": "baseline", "years": 1},
    {"dataset_name": "simple_rules_1y", "generator": "realistic", "scenario": "simple_rules", "years": 1},
    {"dataset_name": "realistic_mixed_1y", "generator": "realistic", "scenario": "realistic_mixed", "years": 1},
    {"dataset_name": "realistic_mixed_2y", "generator": "realistic", "scenario": "realistic_mixed", "years": 2},
    {"dataset_name": "realistic_mixed_3y", "generator": "realistic", "scenario": "realistic_mixed", "years": 3},
    {"dataset_name": "realistic_mixed_5y", "generator": "realistic", "scenario": "realistic_mixed", "years": 5},
    {"dataset_name": "stress_test_1y", "generator": "realistic", "scenario": "stress_test", "years": 1},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supervised and clustering benchmarks across generated datasets")
    parser.add_argument("--num-people", type=int, default=DEFAULT_NUM_PEOPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--annual-inflation-rate", type=float, default=0.025)
    parser.add_argument("--output", type=Path, default=METRICS_DIR / "dataset_comparison.csv")
    return parser.parse_args()


def _generate_dataset(config: dict, num_people: int, seed: int, annual_inflation_rate: float):
    if config["generator"] == "baseline":
        transactions = generate_transactions(num_people=num_people, seed=seed)
        metadata = {
            "missing_log_days": 0,
            "annual_inflation_rate": 0.0,
            "start_date": transactions["date"].min(),
            "end_date": transactions["date"].max(),
        }
        return transactions, metadata

    result = generate_realistic_dataset(
        num_people=num_people,
        years=config["years"],
        scenario=config["scenario"],
        annual_inflation_rate=annual_inflation_rate,
        seed=seed,
    )
    metadata_row = result.generator_metadata.iloc[0].to_dict()
    return result.transactions, metadata_row


def _best_supervised_row(metrics: pd.DataFrame) -> pd.Series:
    model_rows = metrics[~metrics["feature_set"].str.endswith("_baseline")]
    return model_rows.sort_values("test_rmse").iloc[0]


def _best_cluster_row(metrics: pd.DataFrame) -> pd.Series:
    selected_k = int(metrics["selected_k"].iloc[0])
    return metrics[metrics["k"] == selected_k].iloc[0]


def run_comparison(
    num_people: int,
    seed: int,
    annual_inflation_rate: float,
    dataset_matrix: tuple[dict, ...] = DATASET_MATRIX,
) -> pd.DataFrame:
    rows: list[dict] = []
    for config in dataset_matrix:
        print(f"Running {config['dataset_name']}...")
        transactions, metadata = _generate_dataset(config, num_people, seed, annual_inflation_rate)
        daily = aggregate_daily_transactions(transactions)
        supervised_metrics, _, _ = train_supervised_models(daily, seed=seed)
        person_count = int(daily["person_id"].nunique())
        clustering_metrics, assignments = run_clustering(daily, seed=seed, max_k=min(10, person_count - 1))
        best_supervised = _best_supervised_row(supervised_metrics)
        best_cluster = _best_cluster_row(clustering_metrics)

        baseline_rows = supervised_metrics[supervised_metrics["feature_set"].str.endswith("_baseline")]
        best_baseline = baseline_rows.sort_values("test_rmse").iloc[0]
        rows.append(
            {
                "dataset_name": config["dataset_name"],
                "generator": config["generator"],
                "scenario": config["scenario"],
                "years": config["years"],
                "num_people": num_people,
                "transaction_rows": len(transactions),
                "daily_rows": len(daily),
                "missing_log_days": int(metadata.get("missing_log_days", 0)),
                "annual_inflation_rate": float(metadata.get("annual_inflation_rate", 0.0)),
                "date_start": str(metadata.get("start_date", transactions["date"].min())),
                "date_end": str(metadata.get("end_date", transactions["date"].max())),
                "best_supervised_feature_set": best_supervised["feature_set"],
                "best_test_rmse": round(float(best_supervised["test_rmse"]), 6),
                "best_test_mae": round(float(best_supervised["test_mae"]), 6),
                "best_test_r2": round(float(best_supervised["test_r2"]), 6),
                "best_cv_rmse_mean": round(float(best_supervised["cv_rmse_mean"]), 6),
                "best_baseline_feature_set": best_baseline["feature_set"],
                "best_baseline_test_rmse": round(float(best_baseline["test_rmse"]), 6),
                "selected_k": int(assignments["selected_k"].iloc[0]),
                "selected_k_silhouette": round(float(best_cluster["silhouette"]), 6),
                "selected_k_davies_bouldin": round(float(best_cluster["davies_bouldin"]), 6),
                "selected_k_calinski_harabasz": round(float(best_cluster["calinski_harabasz"]), 6),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    ensure_directories()
    comparison = run_comparison(args.num_people, args.seed, args.annual_inflation_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
