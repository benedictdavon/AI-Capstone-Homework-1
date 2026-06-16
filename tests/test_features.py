from expense_modeling.data_generator import generate_transactions
from expense_modeling.features import add_time_series_features, build_model_matrix
from expense_modeling.preprocessing import aggregate_daily_transactions
import pandas as pd


def test_daily_features_are_model_ready():
    transactions = generate_transactions(num_people=5, seed=11)
    daily = aggregate_daily_transactions(transactions)
    X, y, features = build_model_matrix(daily)
    assert len(X) == len(y) == len(daily)
    assert "lag1_expense_ntd" in features
    assert "rolling7_expense_ntd" in features
    assert "rolling30_expense_mean_ntd" in features
    assert "rolling7_meal_expense_sum_ntd" in features
    assert "income_last_30d_ntd" in features
    assert "person_expanding_mean_expense_ntd" in features
    assert not X.isna().any().any()
    assert (y >= 0).all()


def test_rolling_features_do_not_use_same_day_target():
    daily = pd.DataFrame(
        {
            "person_id": ["P001"] * 5,
            "profile_type": ["frugal_student"] * 5,
            "date": pd.date_range("2025-01-01", periods=5),
            "day_of_week": [2, 3, 4, 5, 6],
            "is_weekend": [0, 0, 0, 1, 1],
            "month": [1] * 5,
            "payday_distance": [0, 1, 2, 3, 4],
            "daily_expense_ntd": [100, 200, 300, 400, 500],
            "daily_income_ntd": [1000, 0, 0, 0, 0],
            "net_cash_flow_ntd": [900, -200, -300, -400, -500],
            "category_expense_meal": [10, 20, 30, 40, 50],
        }
    )

    featured = add_time_series_features(daily)

    assert featured.loc[1, "lag1_expense_ntd"] == 100
    assert featured.loc[2, "rolling3_expense_mean_ntd"] == 150
    assert featured.loc[3, "rolling7_meal_expense_sum_ntd"] == 60
    assert featured.loc[4, "person_expanding_mean_expense_ntd"] == 250
