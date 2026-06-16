from expense_modeling.data_generator import generate_transactions
from expense_modeling.features import build_model_matrix
from expense_modeling.preprocessing import aggregate_daily_transactions


def test_daily_features_are_model_ready():
    transactions = generate_transactions(num_people=5, seed=11)
    daily = aggregate_daily_transactions(transactions)
    X, y, features = build_model_matrix(daily)
    assert len(X) == len(y) == len(daily)
    assert "lag1_expense_ntd" in features
    assert "rolling7_expense_ntd" in features
    assert not X.isna().any().any()
    assert (y >= 0).all()
