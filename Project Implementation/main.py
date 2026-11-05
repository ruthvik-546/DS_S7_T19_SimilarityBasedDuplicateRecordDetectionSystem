"""
main.py
-------
End-to-end pipeline matching the project architecture diagram:

  Input Dataset -> Pre-Processing -> Data Splitting -> Feature Representation
  -> Similarity Computation -> Classification/Decision -> Hyperparameter Tuning
  -> Performance Metrics -> Output (duplicate report, cleaned dataset, charts)

Run with:  python3 src/main.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(__file__))

from generate_data import generate_dataset
from preprocessing import preprocess
from feature_engineering import (generate_candidate_pairs, build_pair_features,
                                  add_tfidf_cosine, add_combined_score, label_pairs)
from models import split_data, train_and_evaluate, pick_best_model, FEATURE_COLS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_style("whitegrid")


class UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


def build_cleaned_dataset(df, predicted_duplicate_pairs):
    uf = UnionFind(df["record_id"].tolist())
    for a, b in predicted_duplicate_pairs:
        uf.union(a, b)

    df = df.copy()
    df["cluster_id"] = df["record_id"].apply(uf.find)

    # Keep the record with the fewest missing fields per cluster as the representative
    df["missing_count"] = (df[["name", "email", "phone", "address"]] == "").sum(axis=1)
    cleaned = (df.sort_values("missing_count")
                 .groupby("cluster_id", as_index=False)
                 .first())
    return df, cleaned


def main():
    print("Step 1/6: Generating synthetic input dataset (CSV source)...")
    raw_path = os.path.join(DATA_DIR, "raw_records.csv")
    df_raw = generate_dataset(out_path=raw_path)
    print(f"  -> {len(df_raw)} raw records saved to {raw_path}")

    print("Step 2/6: Pre-processing (cleaning, missing values, normalization, standardization)...")
    df_clean = preprocess(df_raw)

    print("Step 3/6: Generating candidate record pairs (blocking) + feature representation...")
    pairs = generate_candidate_pairs(df_clean)
    pair_df = build_pair_features(df_clean, pairs)
    pair_df = add_tfidf_cosine(df_clean, pair_df)
    pair_df = add_combined_score(pair_df)
    pair_df = label_pairs(pair_df, df_clean)
    print(f"  -> {len(pair_df)} candidate pairs generated "
          f"({pair_df['is_duplicate'].sum()} true duplicate pairs)")

    print("Step 4/6: Splitting data (70% train / 30% test) and training ML models...")
    X_train, X_test, y_train, y_test = split_data(pair_df)
    results, fitted_models = train_and_evaluate(X_train, X_test, y_train, y_test)
    best_name, best_model, best_metrics = pick_best_model(results, fitted_models)
    print(f"  -> Best model: {best_name} (F1 = {best_metrics['f1_score']:.3f})")

    print("Step 5/6: Applying best model to full pair set, building outputs...")
    pair_df["predicted_duplicate"] = best_model.predict(pair_df[FEATURE_COLS])
    duplicate_report = pair_df[pair_df["predicted_duplicate"] == 1].copy()

    rec_lookup = df_clean.set_index("record_id")[["name", "email", "phone", "address"]]
    duplicate_report = duplicate_report.merge(
        rec_lookup.add_prefix("a_"), left_on="id_a", right_index=True
    ).merge(
        rec_lookup.add_prefix("b_"), left_on="id_b", right_index=True
    )
    report_cols = ["id_a", "a_name", "a_email", "a_phone", "a_address",
                    "id_b", "b_name", "b_email", "b_phone", "b_address",
                    "combined_score", "tfidf_cosine", "name_jaro_winkler"]
    duplicate_report = duplicate_report[report_cols].sort_values("combined_score", ascending=False)
    duplicate_report_path = os.path.join(OUT_DIR, "duplicate_records_report.csv")
    duplicate_report.to_csv(duplicate_report_path, index=False)

    predicted_pairs = list(zip(pair_df.loc[pair_df["predicted_duplicate"] == 1, "id_a"],
                                pair_df.loc[pair_df["predicted_duplicate"] == 1, "id_b"]))
    df_with_clusters, cleaned_df = build_cleaned_dataset(df_clean, predicted_pairs)
    cleaned_path = os.path.join(OUT_DIR, "cleaned_unique_dataset.csv")
    cleaned_cols = ["record_id", "name", "email", "phone", "address"]
    cleaned_df[cleaned_cols].to_csv(cleaned_path, index=False)

    print(f"  -> {len(duplicate_report)} record pairs flagged as duplicates -> {duplicate_report_path}")
    print(f"  -> Cleaned dataset: {len(df_clean)} raw -> {len(cleaned_df)} unique records -> {cleaned_path}")

    print("Step 6/6: Generating visualizations and metrics summary...")
    generate_visualizations(pair_df, results, df_clean, cleaned_df)

    metrics_summary = {name: {k: v for k, v in m.items() if k != "confusion_matrix"}
                       for name, m in results.items()}
    for name, m in results.items():
        metrics_summary[name]["confusion_matrix"] = m["confusion_matrix"].tolist()
    metrics_summary["best_model"] = best_name

    metrics_path = os.path.join(OUT_DIR, "model_performance_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  -> Metrics saved to {metrics_path}")

    print("\n=== SUMMARY ===")
    for name, m in results.items():
        print(f"{name:22s} | Acc: {m['accuracy']:.3f}  Prec: {m['precision']:.3f}  "
              f"Recall: {m['recall']:.3f}  F1: {m['f1_score']:.3f}")
    print(f"\nBest model selected: {best_name}")


def generate_visualizations(pair_df, results, df_clean, cleaned_df):
    # 1. Model comparison bar chart
    metrics_names = ["accuracy", "precision", "recall", "f1_score"]
    model_names = list(results.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics_names))
    width = 0.25
    for i, model in enumerate(model_names):
        vals = [results[model][m] for m in metrics_names]
        ax.bar(x + i * width, vals, width, label=model)
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_names])
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)

    # 2. Confusion matrix for best model
    best_name = max(results, key=lambda k: results[k]["f1_score"])
    cm = results[best_name]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Duplicate", "Duplicate"],
                yticklabels=["Non-Duplicate", "Duplicate"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {best_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # 3. Combined similarity score distribution by true label
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(data=pair_df, x="combined_score", hue="is_duplicate",
                 bins=30, kde=True, palette=["#e74c3c", "#2ecc71"], ax=ax)
    ax.set_title("Combined Similarity Score Distribution")
    ax.set_xlabel("Combined Similarity Score")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "similarity_score_distribution.png"), dpi=150)
    plt.close(fig)

    # 4. Dataset size before/after cleaning
    fig, ax = plt.subplots(figsize=(5, 5))
    counts = [len(df_clean), len(cleaned_df)]
    bars = ax.bar(["Raw Records", "Cleaned Unique Records"], counts, color=["#e67e22", "#27ae60"])
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 2, str(count), ha="center")
    ax.set_title("Dataset Size Before vs. After Deduplication")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dataset_size_comparison.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
