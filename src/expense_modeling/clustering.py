from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .features import build_clustering_frame

CLUSTER_FEATURES = [
    "mean_daily_expense_ntd",
    "std_daily_expense_ntd",
    "median_daily_expense_ntd",
    "max_daily_expense_ntd",
    "weekend_expense_mean",
    "weekday_expense_mean",
    "weekend_lift",
    "mean_income_ntd",
    "mean_lag1_expense_ntd",
    "mean_rolling7_expense_ntd",
]


def run_clustering(daily: pd.DataFrame, seed: int = 42, min_k: int = 2, max_k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    person_frame = build_clustering_frame(daily)
    X = person_frame[CLUSTER_FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    metric_rows: list[dict] = []
    models: dict[int, KMeans] = {}
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = model.fit_predict(X_scaled)
        models[k] = model
        metric_rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(X_scaled, labels)),
                "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    best_k = int(metrics.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0]["k"])
    best_kmeans = models[best_k]
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X_scaled)
    gmm = GaussianMixture(n_components=best_k, random_state=seed)
    gmm_labels = gmm.fit_predict(X_scaled)
    gmm_confidence = gmm.predict_proba(X_scaled).max(axis=1)

    assignments = person_frame[["person_id", "profile_type"] + CLUSTER_FEATURES].copy()
    assignments["kmeans_cluster"] = best_kmeans.labels_
    assignments["gmm_cluster"] = gmm_labels
    assignments["gmm_confidence"] = gmm_confidence
    assignments["pca_1"] = coords[:, 0]
    assignments["pca_2"] = coords[:, 1]
    assignments["selected_k"] = best_k
    metrics["selected_k"] = best_k
    metrics["selection_rule"] = "highest silhouette, tie-broken by lowest Davies-Bouldin"
    return metrics, assignments


def save_clustering_outputs(metrics: pd.DataFrame, assignments: pd.DataFrame, metrics_path: Path, assignments_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    assignments.to_csv(assignments_path, index=False)
