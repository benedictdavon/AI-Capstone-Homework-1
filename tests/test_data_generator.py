from expense_modeling.data_generator import generate_transactions


def test_generate_transactions_has_required_schema():
    df = generate_transactions(num_people=8, seed=7)
    required = {
        "person_id",
        "profile_type",
        "date",
        "category",
        "transaction_type",
        "amount_ntd",
        "day_of_week",
        "is_weekend",
        "month",
        "payday_distance",
        "income_source",
        "payment_method",
        "description",
    }
    assert required.issubset(df.columns)
    assert df["person_id"].nunique() == 8
    assert set(df["transaction_type"].unique()) == {"income", "expense"}
    assert (df["amount_ntd"] > 0).all()
