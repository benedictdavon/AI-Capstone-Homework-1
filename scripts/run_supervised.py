from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expense_modeling.config import DAILY_FEATURES_PATH, FEATURE_IMPORTANCE_PATH, FIGURES_DIR, PREDICTIONS_PATH, RAW_TRANSACTIONS_PATH, SUPERVISED_METRICS_PATH, ensure_directories
from expense_modeling.data_generator import generate_transactions, save_transactions
from expense_modeling.preprocessing import aggregate_daily_transactions, load_transactions, save_processed
from expense_modeling.supervised import save_supervised_outputs, train_supervised_models
from expense_modeling.visualization import plot_feature_importance, plot_supervised_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised expense prediction models.")
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
    metrics, importance, predictions = train_supervised_models(daily)
    save_supervised_outputs(metrics, importance, predictions, SUPERVISED_METRICS_PATH, FEATURE_IMPORTANCE_PATH, PREDICTIONS_PATH)
    plot_supervised_metrics(metrics, FIGURES_DIR / "supervised_rmse_by_feature_set.png")
    plot_feature_importance(importance, FIGURES_DIR / "feature_importance_full_behavioral.png")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
