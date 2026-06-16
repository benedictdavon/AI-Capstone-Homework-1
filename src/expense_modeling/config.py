from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
METRICS_DIR = RESULTS_DIR / "metrics"

RAW_TRANSACTIONS_PATH = RAW_DATA_DIR / "synthetic_expense_transactions.csv"
DAILY_FEATURES_PATH = PROCESSED_DATA_DIR / "daily_person_features.csv"
SUPERVISED_METRICS_PATH = METRICS_DIR / "supervised_metrics.csv"
CLUSTER_METRICS_PATH = METRICS_DIR / "clustering_metrics.csv"
CLUSTER_ASSIGNMENTS_PATH = TABLES_DIR / "cluster_assignments.csv"
FEATURE_IMPORTANCE_PATH = TABLES_DIR / "feature_importance.csv"
PREDICTIONS_PATH = TABLES_DIR / "supervised_predictions.csv"

DEFAULT_SEED = 42
DEFAULT_NUM_PEOPLE = 50
DEFAULT_START_DATE = "2025-01-01"
DEFAULT_END_DATE = "2025-12-31"

REQUIRED_DIRECTORIES = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    METRICS_DIR,
]


@dataclass(frozen=True)
class ProfileConfig:
    profile_type: str
    weight: int
    daily_meal_multiplier: float
    coffee_probability: float
    weekend_social_probability: float
    transport_multiplier: float
    income_multiplier: float
    variance_multiplier: float
    gym_probability: float
    part_time_probability: float


PROFILE_CONFIGS: tuple[ProfileConfig, ...] = (
    ProfileConfig("frugal_student", 9, 0.82, 0.18, 0.08, 0.85, 0.95, 0.70, 0.05, 0.25),
    ProfileConfig("food_heavy_spender", 8, 1.35, 0.65, 0.20, 1.00, 1.00, 1.00, 0.10, 0.30),
    ProfileConfig("commuter", 7, 1.00, 0.35, 0.12, 1.80, 1.00, 0.90, 0.05, 0.30),
    ProfileConfig("social_weekend_spender", 8, 1.05, 0.40, 0.58, 1.10, 1.00, 1.15, 0.10, 0.35),
    ProfileConfig("irregular_high_variance_spender", 7, 1.10, 0.35, 0.25, 1.10, 1.05, 2.10, 0.12, 0.35),
    ProfileConfig("part_time_worker", 7, 1.00, 0.32, 0.20, 1.05, 1.30, 1.00, 0.08, 0.85),
    ProfileConfig("lifestyle_gym_spender", 4, 1.18, 0.45, 0.22, 1.00, 1.05, 1.20, 0.55, 0.35),
)


def ensure_directories() -> None:
    for path in REQUIRED_DIRECTORIES:
        path.mkdir(parents=True, exist_ok=True)
