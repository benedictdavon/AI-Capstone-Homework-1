from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expense_modeling.config import DEFAULT_NUM_PEOPLE, DEFAULT_SEED, DEFAULT_START_DATE, RAW_DATA_DIR, ensure_directories
from expense_modeling.data_generator_realistic import DURATION_YEARS, SCENARIOS, generate_realistic_dataset, save_realistic_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate realistic synthetic personal finance transactions")
    parser.add_argument("--num-people", type=int, default=DEFAULT_NUM_PEOPLE)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None, help="Optional explicit end date. Overrides --years when provided")
    parser.add_argument("--years", type=int, default=1, choices=DURATION_YEARS)
    parser.add_argument("--scenario", default="realistic_mixed", choices=SCENARIOS)
    parser.add_argument("--annual-inflation-rate", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    result = generate_realistic_dataset(
        num_people=args.num_people,
        start_date=args.start_date,
        end_date=args.end_date,
        years=args.years,
        scenario=args.scenario,
        annual_inflation_rate=args.annual_inflation_rate,
        seed=args.seed,
    )
    paths = save_realistic_dataset(result, args.output_dir)
    metadata = result.generator_metadata.iloc[0]
    print(
        f"Generated {metadata['transaction_rows']:,} transactions for {args.num_people} people "
        f"from {metadata['start_date']} to {metadata['end_date']}"
    )
    print(f"Scenario: {args.scenario}")
    print(f"Missing log days: {metadata['missing_log_days']:,}")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
