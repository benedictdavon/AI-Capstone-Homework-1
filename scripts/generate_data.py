from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expense_modeling.config import DEFAULT_END_DATE, DEFAULT_NUM_PEOPLE, DEFAULT_SEED, DEFAULT_START_DATE, RAW_TRANSACTIONS_PATH, ensure_directories
from expense_modeling.data_generator import generate_transactions, save_transactions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic multi-person expense transactions.")
    parser.add_argument("--num-people", type=int, default=DEFAULT_NUM_PEOPLE)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=RAW_TRANSACTIONS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    transactions = generate_transactions(args.num_people, args.start_date, args.end_date, args.seed)
    output = save_transactions(transactions, args.output)
    print(f"Generated {len(transactions):,} transactions for {transactions['person_id'].nunique()} people: {output}")
    print(transactions.groupby('profile_type')['person_id'].nunique().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
