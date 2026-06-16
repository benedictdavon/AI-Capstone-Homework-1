from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_supervised_metrics(metrics: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = metrics.sort_values("test_rmse")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(ordered["feature_set"], ordered["test_rmse"], color="#4C78A8")
    ax.set_title("Random Forest Test RMSE by Feature Set")
    ax.set_xlabel("Feature set")
    ax.set_ylabel("RMSE (NTD)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_feature_importance(importance: pd.DataFrame, output_path: str | Path, feature_set: str = "full_behavioral") -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    subset = importance[importance["feature_set"] == feature_set].sort_values("importance", ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(subset["feature"], subset["importance"], color="#59A14F")
    ax.set_title(f"Top Feature Importances ({feature_set})")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_clustering_metrics(metrics: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(metrics["k"], metrics["silhouette"], marker="o", label="Silhouette", color="#4C78A8")
    ax2 = ax.twinx()
    ax2.plot(metrics["k"], metrics["davies_bouldin"], marker="s", label="Davies-Bouldin", color="#E15759")
    ax.set_title("K-Means Cluster Selection Metrics")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette (higher is better)")
    ax2.set_ylabel("Davies-Bouldin (lower is better)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_cluster_pca(assignments: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.4))
    clusters = sorted(assignments["kmeans_cluster"].unique())
    for cluster in clusters:
        subset = assignments[assignments["kmeans_cluster"] == cluster]
        ax.scatter(subset["pca_1"], subset["pca_2"], s=70, label=f"cluster {cluster}", alpha=0.85)
    ax.set_title("Person-Level Spending Clusters (PCA Projection)")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
