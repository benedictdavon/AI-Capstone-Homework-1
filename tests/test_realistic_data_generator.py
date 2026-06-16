from expense_modeling.data_generator_realistic import generate_realistic_dataset


def test_realistic_generator_outputs_required_files_data():
    result = generate_realistic_dataset(num_people=10, years=1, seed=21)
    transactions = result.transactions
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
    assert required.issubset(transactions.columns)
    assert transactions["person_id"].nunique() == 10
    assert set(transactions["transaction_type"].unique()) == {"income", "expense"}
    assert (transactions["amount_ntd"] > 0).all()
    assert len(result.person_profiles) == 10
    assert result.day_metadata["missing_log_day"].sum() > 0


def test_realistic_generator_is_reproducible_for_same_seed():
    first = generate_realistic_dataset(num_people=6, years=1, seed=5)
    second = generate_realistic_dataset(num_people=6, years=1, seed=5)
    assert first.transactions.equals(second.transactions)
    assert first.person_profiles.equals(second.person_profiles)
    assert first.day_metadata.equals(second.day_metadata)


def test_realistic_generator_supports_multi_year_ranges():
    result = generate_realistic_dataset(num_people=3, years=3, seed=8)
    metadata = result.generator_metadata.iloc[0]
    assert metadata["start_date"] == "2025-01-01"
    assert metadata["end_date"] == "2027-12-31"
    assert result.day_metadata["date"].min() == "2025-01-01"
    assert result.day_metadata["date"].max() == "2027-12-31"


def test_stress_test_has_more_variance_than_simple_rules():
    simple = generate_realistic_dataset(num_people=30, years=1, scenario="simple_rules", seed=42)
    stress = generate_realistic_dataset(num_people=30, years=1, scenario="stress_test", seed=42)
    simple_expense_std = simple.transactions[simple.transactions["transaction_type"] == "expense"]["amount_ntd"].std()
    stress_expense_std = stress.transactions[stress.transactions["transaction_type"] == "expense"]["amount_ntd"].std()
    assert stress_expense_std > simple_expense_std
