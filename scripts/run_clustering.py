from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expense_modeling.clustering import run_clustering, save_clustering_outputs
from expense_modeling.config import CLUSTER_ASSIGNMENTS_PATH, CLUSTER_METRICS_PATH, DAILY_FEATURES_PATH, FIGURES_DIR, RAW_TRANSACTIONS_PATH, ensure_directories
from expense_modeling.data_generator import generate_transactions, save_transactions
from expense_modeling.preprocessing import aggregate_daily_transactions, load_transactions, save_processed
from expense_modeling.visualization import plot_cluster_pca, plot_clustering_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run person-level spending clustering.")
    parser.add_argument("--transactions", type=Path, default=RAW_TRANSACTIONS_PATH)
    parser.add_argument("--generate-if-missing", action="store_true", help="Generate default synthetic data if the transaction file is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    if not args.transactions.exists():
        if not args.generate_if_missing:
            raise FileNotFoundError(f"Missing {args.transactions}. Run scripts/generate_data.py first or pass --generate-if-missing.")
        save_transactions(generate_transactions(), args.transactions)
    transactions = load_transactions(args.transactions)
    daily = aggregate_daily_transactions(transactions)
    save_processed(daily, DAILY_FEATURES_PATH)
    metrics, assignments = run_clustering(daily)
    save_clustering_outputs(metrics, assignments, CLUSTER_METRICS_PATH, CLUSTER_ASSIGNMENTS_PATH)
    plot_clustering_metrics(metrics, FIGURES_DIR / "clustering_metrics.png")
    plot_cluster_pca(assignments, FIGURES_DIR / "spending_clusters_pca.png")
    print(metrics.to_string(index=False))
    print(f"Selected k: {int(assignments['selected_k'].iloc[0])}")


if __name__ == "__main__":
    main()
