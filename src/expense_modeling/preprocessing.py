from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "person_id",
    "profile_type",
    "date",
    "category",
    "transaction_type",
    "amount_ntd",
    "day_of_week",
    "is_weekend",
    "month",
}


def load_transactions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required transaction columns: {sorted(missing)}")
    return df


def aggregate_daily_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()
    df["signed_amount"] = df["amount_ntd"].where(df["transaction_type"] == "income", -df["amount_ntd"])
    expense = (
        df[df["transaction_type"] == "expense"]
        .groupby(["person_id", "profile_type", "date"], as_index=False)["amount_ntd"]
        .sum()
        .rename(columns={"amount_ntd": "daily_expense_ntd"})
    )
    income = (
        df[df["transaction_type"] == "income"]
        .groupby(["person_id", "profile_type", "date"], as_index=False)["amount_ntd"]
        .sum()
        .rename(columns={"amount_ntd": "daily_income_ntd"})
    )
    calendar = df[["person_id", "profile_type", "date", "day_of_week", "is_weekend", "month", "payday_distance"]].drop_duplicates()
    daily = calendar.merge(expense, on=["person_id", "profile_type", "date"], how="left")
    daily = daily.merge(income, on=["person_id", "profile_type", "date"], how="left")
    daily[["daily_expense_ntd", "daily_income_ntd"]] = daily[["daily_expense_ntd", "daily_income_ntd"]].fillna(0)
    daily["net_cash_flow_ntd"] = daily["daily_income_ntd"] - daily["daily_expense_ntd"]
    return daily.sort_values(["person_id", "date"]).reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output
