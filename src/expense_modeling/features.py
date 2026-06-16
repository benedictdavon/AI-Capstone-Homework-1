from __future__ import annotations

import pandas as pd

BASE_NUMERIC_FEATURES = ["day_of_week", "is_weekend", "month", "payday_distance"]
TIME_FEATURES = ["lag1_expense_ntd", "rolling7_expense_ntd"]
TARGET_COLUMN = "daily_expense_ntd"


def add_time_series_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.sort_values(["person_id", "date"]).copy()
    grouped = df.groupby("person_id", group_keys=False)
    df["lag1_expense_ntd"] = grouped["daily_expense_ntd"].shift(1)
    df["rolling7_expense_ntd"] = grouped["daily_expense_ntd"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=1).mean()
    )
    df["lag1_expense_ntd"] = df["lag1_expense_ntd"].fillna(df.groupby("person_id")["daily_expense_ntd"].transform("median"))
    df["rolling7_expense_ntd"] = df["rolling7_expense_ntd"].fillna(df["lag1_expense_ntd"])
    return df


def build_model_matrix(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = add_time_series_features(daily)
    profile_dummies = pd.get_dummies(df["profile_type"], prefix="profile", dtype=int)
    person_dummies = pd.get_dummies(df["person_id"], prefix="person", dtype=int)
    X = pd.concat(
        [
            df[BASE_NUMERIC_FEATURES + TIME_FEATURES].reset_index(drop=True),
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
