from __future__ import annotations

import pandas as pd

BASE_NUMERIC_FEATURES = [
    "day_of_week",
    "is_weekend",
    "month",
    "payday_distance",
]
CALENDAR_FEATURES = [
    "day_of_month",
    "week_of_year",
    "is_month_start",
    "is_month_end",
]
TIME_FEATURES = ["lag1_expense_ntd", "rolling7_expense_ntd"]
ROLLING_WINDOWS = (3, 7, 14, 30)
ROLLING_STATS = ("mean", "std", "max", "min")
CATEGORY_ROLLING_WINDOWS = (7, 14, 30)
CATEGORY_ROLLING_CATEGORIES = (
    "meal",
    "coffee",
    "transport",
    "groceries",
    "social_food",
    "gym",
    "irregular_purchase",
)
INCOME_TIMING_FEATURES = [
    "days_since_last_income",
    "income_last_7d_ntd",
    "income_last_14d_ntd",
    "income_last_30d_ntd",
]
BASELINE_FEATURES = [
    "person_expanding_mean_expense_ntd",
    "person_expanding_median_expense_ntd",
    "person_weekend_expanding_mean_expense_ntd",
    "person_weekday_expanding_mean_expense_ntd",
    "profile_expanding_mean_expense_ntd",
]
INTERACTION_FEATURES = [
    "is_weekend_profile_frugal_student",
    "is_weekend_profile_food_heavy_spender",
    "is_weekend_profile_commuter",
    "is_weekend_profile_social_weekend_spender",
    "is_weekend_profile_irregular_high_variance_spender",
    "is_weekend_profile_part_time_worker",
    "is_weekend_profile_lifestyle_gym_spender",
]
TARGET_COLUMN = "daily_expense_ntd"


def _safe_expanding_mean(values: pd.Series) -> pd.Series:
    return values.shift(1).expanding(min_periods=1).mean()


def _safe_expanding_median(values: pd.Series) -> pd.Series:
    return values.shift(1).expanding(min_periods=1).median()


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["day_of_month"] = enriched["date"].dt.day
    enriched["week_of_year"] = enriched["date"].dt.isocalendar().week.astype(int)
    enriched["is_month_start"] = enriched["date"].dt.is_month_start.astype(int)
    enriched["is_month_end"] = enriched["date"].dt.is_month_end.astype(int)
    return enriched


def _add_person_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    grouped = enriched.groupby("person_id", group_keys=False)
    shifted_expense = grouped["daily_expense_ntd"].shift(1)
    enriched["lag1_expense_ntd"] = shifted_expense
    enriched["rolling7_expense_ntd"] = grouped["daily_expense_ntd"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=1).mean()
    )
    for window in ROLLING_WINDOWS:
        rolling = shifted_expense.groupby(enriched["person_id"]).rolling(window, min_periods=1)
        enriched[f"rolling{window}_expense_mean_ntd"] = rolling.mean().reset_index(level=0, drop=True)
        enriched[f"rolling{window}_expense_std_ntd"] = rolling.std().reset_index(level=0, drop=True)
        enriched[f"rolling{window}_expense_max_ntd"] = rolling.max().reset_index(level=0, drop=True)
        enriched[f"rolling{window}_expense_min_ntd"] = rolling.min().reset_index(level=0, drop=True)
    return enriched


def _add_category_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    for category in CATEGORY_ROLLING_CATEGORIES:
        source_column = f"category_expense_{category}"
        if source_column not in enriched.columns:
            enriched[source_column] = 0
        shifted_category = enriched.groupby("person_id")[source_column].shift(1)
        for window in CATEGORY_ROLLING_WINDOWS:
            feature_name = f"rolling{window}_{category}_expense_sum_ntd"
            enriched[feature_name] = (
                shifted_category.groupby(enriched["person_id"]).rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
            )
    return enriched


def _add_income_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["has_income"] = enriched["daily_income_ntd"] > 0
    enriched["income_date"] = enriched["date"].where(enriched["has_income"])
    last_income_date = enriched.groupby("person_id")["income_date"].ffill().groupby(enriched["person_id"]).shift(1)
    enriched["days_since_last_income"] = (enriched["date"] - last_income_date).dt.days
    shifted_income = enriched.groupby("person_id")["daily_income_ntd"].shift(1)
    for window in (7, 14, 30):
        enriched[f"income_last_{window}d_ntd"] = (
            shifted_income.groupby(enriched["person_id"]).rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
        )
    return enriched.drop(columns=["has_income", "income_date"])


def _add_expanding_baselines(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    person_grouped = enriched.groupby("person_id", group_keys=False)
    enriched["person_expanding_mean_expense_ntd"] = person_grouped["daily_expense_ntd"].transform(_safe_expanding_mean)
    enriched["person_expanding_median_expense_ntd"] = person_grouped["daily_expense_ntd"].transform(_safe_expanding_median)

    is_weekend = enriched["is_weekend"] == 1
    weekend_expense = enriched["daily_expense_ntd"].where(is_weekend)
    weekday_expense = enriched["daily_expense_ntd"].where(~is_weekend)
    enriched["person_weekend_expanding_mean_expense_ntd"] = weekend_expense.groupby(enriched["person_id"]).transform(
        _safe_expanding_mean
    )
    enriched["person_weekday_expanding_mean_expense_ntd"] = weekday_expense.groupby(enriched["person_id"]).transform(
        _safe_expanding_mean
    )

    profile_daily = (
        enriched.groupby(["profile_type", "date"], as_index=False)["daily_expense_ntd"]
        .mean()
        .sort_values(["profile_type", "date"])
    )
    profile_daily["profile_expanding_mean_expense_ntd"] = profile_daily.groupby("profile_type")["daily_expense_ntd"].transform(
        _safe_expanding_mean
    )
    enriched = enriched.merge(
        profile_daily[["profile_type", "date", "profile_expanding_mean_expense_ntd"]],
        on=["profile_type", "date"],
        how="left",
    )
    enriched["person_weekend_or_weekday_baseline_ntd"] = enriched["person_weekday_expanding_mean_expense_ntd"].where(
        ~is_weekend, enriched["person_weekend_expanding_mean_expense_ntd"]
    )
    return enriched


def _add_profile_interactions(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    for column in INTERACTION_FEATURES:
        profile_type = column.removeprefix("is_weekend_profile_")
        enriched[column] = ((enriched["profile_type"] == profile_type) & (enriched["is_weekend"] == 1)).astype(int)
    return enriched


def _fill_feature_defaults(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    model_columns = get_numeric_feature_columns(enriched)
    enriched[model_columns] = enriched[model_columns].fillna(0)
    return enriched


def add_time_series_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.sort_values(["person_id", "date"]).copy()
    df = _add_calendar_features(df)
    df = _add_person_rolling_features(df)
    df = _add_category_rolling_features(df)
    df = _add_income_timing_features(df)
    df = _add_expanding_baselines(df)
    df = _add_profile_interactions(df)
    return _fill_feature_defaults(df)


def get_rolling_feature_columns() -> list[str]:
    return [
        f"rolling{window}_expense_{stat}_ntd"
        for window in ROLLING_WINDOWS
        for stat in ROLLING_STATS
    ]


def get_category_rolling_feature_columns() -> list[str]:
    return [
        f"rolling{window}_{category}_expense_sum_ntd"
        for category in CATEGORY_ROLLING_CATEGORIES
        for window in CATEGORY_ROLLING_WINDOWS
    ]


def get_advanced_feature_columns() -> list[str]:
    return (
        BASE_NUMERIC_FEATURES
        + CALENDAR_FEATURES
        + TIME_FEATURES
        + get_rolling_feature_columns()
        + get_category_rolling_feature_columns()
        + INCOME_TIMING_FEATURES
        + BASELINE_FEATURES
        + ["person_weekend_or_weekday_baseline_ntd"]
        + INTERACTION_FEATURES
    )


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    known_features = set(get_advanced_feature_columns())
    category_expense_columns = {column for column in df.columns if column.startswith("category_expense_")}
    return [column for column in df.columns if column in known_features or column in category_expense_columns]


def build_model_matrix(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = add_time_series_features(daily)
    profile_dummies = pd.get_dummies(df["profile_type"], prefix="profile", dtype=int)
    person_dummies = pd.get_dummies(df["person_id"], prefix="person", dtype=int)
    numeric_features = get_numeric_feature_columns(df)
    X = pd.concat(
        [
            df[numeric_features].reset_index(drop=True),
            profile_dummies.reset_index(drop=True),
            person_dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    y = df[TARGET_COLUMN].reset_index(drop=True)
    return X, y, list(X.columns)


def build_clustering_frame(daily: pd.DataFrame) -> pd.DataFrame:
    df = add_time_series_features(daily)
    summary = (
        df.groupby(["person_id", "profile_type"], as_index=False)
        .agg(
            mean_daily_expense_ntd=("daily_expense_ntd", "mean"),
            std_daily_expense_ntd=("daily_expense_ntd", "std"),
            median_daily_expense_ntd=("daily_expense_ntd", "median"),
            max_daily_expense_ntd=("daily_expense_ntd", "max"),
            weekend_expense_mean=("daily_expense_ntd", lambda x: x[df.loc[x.index, "is_weekend"] == 1].mean()),
            weekday_expense_mean=("daily_expense_ntd", lambda x: x[df.loc[x.index, "is_weekend"] == 0].mean()),
            mean_income_ntd=("daily_income_ntd", "mean"),
            mean_lag1_expense_ntd=("lag1_expense_ntd", "mean"),
            mean_rolling7_expense_ntd=("rolling7_expense_ntd", "mean"),
        )
    )
    summary["weekend_lift"] = summary["weekend_expense_mean"] - summary["weekday_expense_mean"]
    return summary.fillna(0)
