#!/usr/bin/env python
"""
Make a zoomed RFIC v1 scatter plot from saved JSON results.
The normal plot can hide tiny positive B values when bad candidates have much
larger negative B. This view centers the y-axis around B=0 so small local wins
above the zero line are visible.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weight_handles.uo import layer_slug


DEFAULT_LAYERS = [
    "model.layers.7.mlp.down_proj",
    "model.layers.14.mlp.down_proj",
    "model.layers.20.mlp.down_proj",
]
RULES = ["ruleA", "ruleB", "ruleC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make zoomed RFIC v1 B≈0 scatter plot")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="results/figures/uo_v1_all_layers_all_rules_zoom.png")
    parser.add_argument("--layers", nargs="*", default=DEFAULT_LAYERS)
    return parser.parse_args()


def read_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_results(results_dir: Path, layers: list[str]) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, list[dict[str, Any]]]]:
    all_results = {}
    baselines = {}
    for layer_name in layers:
        slug = layer_slug(layer_name)
        all_results[layer_name] = {}
        baselines[layer_name] = read_json(results_dir / f"uo_v1_{slug}_v0_top_abs.json")
        for rule in RULES:
            all_results[layer_name][rule] = read_json(results_dir / f"uo_v1_{slug}_{rule}.json")
    return all_results, baselines


def x_limits(all_results: dict[str, dict[str, list[dict[str, Any]]]], baselines: dict[str, list[dict[str, Any]]]) -> tuple[float, float]:
    xs = []
    for layer_records in all_results.values():
        for records in layer_records.values():
            xs.extend(max(row["A_local_g"], 1e-10) for row in records)
    for records in baselines.values():
        xs.extend(max(row["A_local_g"], 1e-10) for row in records)
    if not xs:
        return 1e-10, 1.0
    return min(xs) * 0.8, max(xs) * 1.25


def zoom_y_limits(all_results: dict[str, dict[str, list[dict[str, Any]]]]) -> tuple[float, float]:
    positives = []
    near_negatives = []
    for layer_records in all_results.values():
        for records in layer_records.values():
            for row in records:
                b_value = row["B_g"]
                if b_value > 0:
                    positives.append(b_value)
                else:
                    near_negatives.append(abs(b_value))

    scale = max(positives) if positives else 1e-8
    # explanation: we intentionally clip away very negative failures here; the
    # purpose of this figure is to inspect whether any dots barely cross above
    # B=0, not to show the full damage range.
    return -1.25 * scale, 1.25 * scale


def draw_panel(ax: Any, records: list[dict[str, Any]], baseline: list[dict[str, Any]], title: str) -> None:
    if baseline:
        # explanation: gray v0 points show what the old "largest weight" rule did,
        # so the zoomed plot can show whether v1 actually moved candidates upward.
        ax.scatter(
            [max(row["A_local_g"], 1e-10) for row in baseline],
            [row["B_g"] for row in baseline],
            c="gray",
            alpha=0.2,
            edgecolors="none",
            label="v0 top |w|",
        )
    top = sorted(records, key=lambda row: row["score_g"], reverse=True)[:10]
    top_keys = {(row["group"], row["row"], row["col"]) for row in top}
    colors = ["red" if (row["group"], row["row"], row["col"]) in top_keys else "steelblue" for row in records]
    ax.scatter(
        [max(row["A_local_g"], 1e-10) for row in records],
        [row["B_g"] for row in records],
        c=colors,
        alpha=0.85,
        edgecolors="none",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    all_results, baselines = load_results(results_dir, list(args.layers))
    xlim = x_limits(all_results, baselines)
    ylim = zoom_y_limits(all_results)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(args.layers)
    fig, axes = plt.subplots(len(layers), len(RULES), figsize=(15, 4 * len(layers)), sharex=True, sharey=True)
    for i, layer_name in enumerate(layers):
        for j, rule in enumerate(RULES):
            ax = axes[i][j] if len(layers) > 1 else axes[j]
            draw_panel(ax, all_results[layer_name][rule], baselines[layer_name], f"{layer_slug(layer_name)} {rule}")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if j == 0:
                ax.set_ylabel("B near zero")
            if i == len(layers) - 1:
                ax.set_xlabel("A_local damage")

    fig.suptitle("RFIC v1 zoom: small positive B values above the zero line")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"saved zoomed figure: {output}")


if __name__ == "__main__":
    main()
