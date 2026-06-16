from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import regression_metrics
from .features import BASE_NUMERIC_FEATURES, TIME_FEATURES, build_model_matrix

FEATURE_SETS = {
    "calendar_only": BASE_NUMERIC_FEATURES,
    "calendar_lag": BASE_NUMERIC_FEATURES + ["lag1_expense_ntd"],
    "calendar_rolling": BASE_NUMERIC_FEATURES + ["rolling7_expense_ntd"],
    "full_behavioral": BASE_NUMERIC_FEATURES + TIME_FEATURES,
}


def _time_based_split(frame: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.Series, pd.Series]:
    cutoff = frame["date"].quantile(1 - test_fraction)
    train_mask = frame["date"] <= cutoff
    test_mask = frame["date"] > cutoff
    return train_mask, test_mask


def train_supervised_models(daily: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_all, y, all_features = build_model_matrix(daily)
    frame = daily.sort_values(["person_id", "date"]).reset_index(drop=True)
    X_all = X_all.loc[frame.index].reset_index(drop=True)
    y = y.loc[frame.index].reset_index(drop=True)
    profile_features = [column for column in all_features if column.startswith("profile_")]
    person_features = [column for column in all_features if column.startswith("person_")]
    train_mask, test_mask = _time_based_split(frame)

    metrics_rows: list[dict] = []
    importance_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for feature_set, numeric_features in FEATURE_SETS.items():
        selected_features = numeric_features + profile_features + person_features
        model = RandomForestRegressor(
            n_estimators=250,
            max_depth=16,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
        X_train = X_all.loc[train_mask, selected_features]
        y_train = y.loc[train_mask]
        X_test = X_all.loc[test_mask, selected_features]
        y_test = y.loc[test_mask]
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = regression_metrics(y_test, predictions)

        cv = TimeSeriesSplit(n_splits=5)
        cv_scores = -cross_val_score(
            pipeline,
            X_all[selected_features],
            y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=None,
        )
        metrics_rows.append(
            {
                "feature_set": feature_set,
                "num_features": len(selected_features),
                "test_rmse": metrics["rmse"],
                "test_mae": metrics["mae"],
                "test_r2": metrics["r2"],
                "cv_rmse_mean": float(cv_scores.mean()),
                "cv_rmse_std": float(cv_scores.std()),
            }
        )

        fitted_model = pipeline.named_steps["model"]
        for feature, importance in zip(selected_features, fitted_model.feature_importances_):
            importance_rows.append({"feature_set": feature_set, "feature": feature, "importance": float(importance)})

        prediction_frame = frame.loc[test_mask, ["person_id", "profile_type", "date", "daily_expense_ntd"]].copy()
        prediction_frame["feature_set"] = feature_set
        prediction_frame["predicted_daily_expense_ntd"] = predictions
        prediction_frames.append(prediction_frame)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("test_rmse").reset_index(drop=True)
    importance_df = pd.DataFrame(importance_rows).sort_values(["feature_set", "importance"], ascending=[True, False])
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    return metrics_df, importance_df, predictions_df


def save_supervised_outputs(metrics: pd.DataFrame, importance: pd.DataFrame, predictions: pd.DataFrame, metrics_path: Path, importance_path: Path, predictions_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    importance.to_csv(importance_path, index=False)
    predictions.to_csv(predictions_path, index=False)
