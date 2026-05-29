#!/usr/bin/env python
"""
Create taxonomy tables and summary figures from weight-outlier results.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weight_handles.taxonomy import classify_results, taxonomy_counts


def parse_args():
    parser = argparse.ArgumentParser(description="Build weight-outlier report artifacts")
    parser.add_argument("--input", default="results/feature_handles.json")
    parser.add_argument("--table", default="results/tables/feature_handle_taxonomy.csv")
    parser.add_argument("--figure", default="results/figures/feature_handle_taxonomy.png")
    return parser.parse_args()


def main():
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(args.input, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    candidates = payload.get("candidates", payload)
    classified = classify_results(candidates)

    table_path = Path(args.table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "taxonomy",
            "source",
            "layer",
            "matrix_type",
            "row",
            "col",
            "value",
            "score",
            "mean_kl",
            "tail_cvar_kl",
            "top_token_flip_rate",
            "delta_ppl",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in classified:
            ablation = item.get("ablation", {})
            writer.writerow({
                "taxonomy": item.get("taxonomy"),
                "source": item.get("source"),
                "layer": item.get("layer"),
                "matrix_type": item.get("matrix_type"),
                "row": item.get("row"),
                "col": item.get("col"),
                "value": item.get("value"),
                "score": item.get("score"),
                "mean_kl": ablation.get("mean_kl"),
                "tail_cvar_kl": ablation.get("tail_cvar_kl"),
                "top_token_flip_rate": ablation.get("top_token_flip_rate"),
                "delta_ppl": ablation.get("delta_ppl"),
            })

    counts = taxonomy_counts(classified)
    figure_path = Path(args.figure)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_ylabel("Candidates")
    ax.set_title("Weight-Outlier Taxonomy")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)

    print(f"Saved table to {table_path}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
