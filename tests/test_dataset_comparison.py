from scripts.run_dataset_comparison import DATASET_MATRIX, run_comparison


def test_dataset_comparison_schema_for_small_run():
    small_matrix = (
        DATASET_MATRIX[0],
        DATASET_MATRIX[2],
    )
    comparison = run_comparison(num_people=6, seed=13, annual_inflation_rate=0.025, dataset_matrix=small_matrix)
    required = {
        "dataset_name",
        "scenario",
        "years",
        "best_supervised_feature_set",
        "best_test_rmse",
        "best_test_mae",
        "best_test_r2",
        "best_baseline_feature_set",
        "selected_k",
        "selected_k_silhouette",
        "missing_log_days",
    }
    assert required.issubset(comparison.columns)
    assert len(comparison) == len(small_matrix)
    assert (comparison["best_test_rmse"] > 0).all()
