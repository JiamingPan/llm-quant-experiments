#!/usr/bin/env python
"""
RFIC v1 tests smarter Under-Outlier candidate rules on several Qwen3 MLP layers.
Each rule still proposes one scalar per quantization group, but now candidates
must combine large weight magnitude with weak activation support. Upper-left
scatter dots mean low FP16 damage and positive local quantization benefit.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awq.core import collect_linear_inputs
from weight_handles.strata import flatten_strata, load_strata
from weight_handles.uo import evaluate_candidate, layer_slug, top_abs_candidate


MODEL_PATH = "/scratch/huterer_root/huterer0/jiamingp/models/qwen3-1b7"
DEFAULT_LAYERS = [
    "model.layers.7.mlp.down_proj",
    "model.layers.14.mlp.down_proj",
    "model.layers.20.mlp.down_proj",
]
RULES = ["ruleA", "ruleB", "ruleC"]
N_BITS = 4
GROUP_SIZE = 128
ETA = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RFIC v1 smarter UO candidate rules")
    parser.add_argument("--config", default="configs/uo_v1.yaml")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--layers", nargs="*", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--combined-fig", default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_settings(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    run_cfg = cfg.get("rfic_v1", {})
    return {
        "model_name": args.model_name or cfg.get("model", {}).get("name", MODEL_PATH),
        "layers": args.layers or run_cfg.get("layers", DEFAULT_LAYERS),
        "max_tokens": args.max_tokens or run_cfg.get("max_tokens", 512),
        "max_length": args.max_length or run_cfg.get("max_length", 512),
        "output_dir": args.output_dir or run_cfg.get("output_dir", "results"),
        "figure_dir": args.figure_dir or run_cfg.get("figure_dir", "results/figures"),
        "summary_csv": args.summary_csv or run_cfg.get("summary_csv", "results/uo_v1_summary.csv"),
        "combined_fig": args.combined_fig or run_cfg.get(
            "combined_fig", "results/figures/uo_v1_all_layers_all_rules.png"
        ),
    }


def load_model(model_name: str) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def group_slice(group_index: int, group_size: int = GROUP_SIZE) -> slice:
    return slice(group_index * group_size, (group_index + 1) * group_size)


def candidate_rule_a(W_group: torch.Tensor, act_group: torch.Tensor) -> tuple[int, int]:
    assert W_group.ndim == 2
    assert act_group.ndim == 1
    assert W_group.shape[1] == act_group.shape[0]

    abs_w = W_group.abs()
    k = max(1, math.ceil(0.25 * abs_w.numel()))
    top_idx = torch.topk(abs_w.reshape(-1), k=k).indices
    # explanation: scalar path importance is roughly weight^2 times activation
    # support times downstream reader gain. We do not know reader gain in v1, so
    # |w| * mean|a| is the simplest "large but quiet input channel" proxy.
    proxy = (abs_w * act_group.view(1, -1)).reshape(-1)
    # explanation: UOs are supposed to be large weights with low function, so
    # Rule A only searches inside the top-25% magnitude subset instead of all weights.
    best = int(top_idx[torch.argmin(proxy[top_idx])].item())
    return best // W_group.shape[1], best % W_group.shape[1]


def candidate_rule_b(W_group: torch.Tensor, dead_local_cols: torch.Tensor) -> tuple[int, int] | None:
    assert W_group.ndim == 2
    assert dead_local_cols.ndim == 1
    if dead_local_cols.numel() == 0:
        return None

    subset = W_group[:, dead_local_cols].abs()
    flat = int(torch.argmax(subset).item())
    row = flat // dead_local_cols.numel()
    local_col = int(dead_local_cols[flat % dead_local_cols.numel()].item())
    return row, local_col


def candidate_rule_c(W_group: torch.Tensor, below_median_cols: torch.Tensor) -> tuple[int, int] | None:
    assert W_group.ndim == 2
    assert below_median_cols.ndim == 1
    if below_median_cols.numel() == 0:
        return None

    group_median = torch.median(W_group.abs())
    scores = W_group[:, below_median_cols].abs() - group_median
    flat = int(torch.argmax(scores).item())
    row = flat // below_median_cols.numel()
    local_col = int(below_median_cols[flat % below_median_cols.numel()].item())
    return row, local_col


def select_candidates_for_layer(W: torch.Tensor, X: torch.Tensor) -> tuple[dict[str, list[dict[str, int]]], list[dict[str, int]]]:
    assert W.ndim == 2
    assert X.ndim == 2
    assert X.shape[1] == W.shape[1]
    assert W.shape[1] % GROUP_SIZE == 0

    # explanation: these calibration activations tell us which input channels
    # are quiet on real data; Rule B still needs them because a "dead" channel is
    # defined by mean activation, not by the weight matrix alone.
    act = X.abs().mean(dim=0)
    dead_count = max(1, math.ceil(0.05 * act.numel()))
    dead_cols = set(int(idx.item()) for idx in torch.topk(-act, k=dead_count).indices)
    act_median = torch.median(act)
    n_groups = W.shape[1] // GROUP_SIZE
    selected = {rule: [] for rule in RULES}
    v0 = []

    for group in range(n_groups):
        cols = group_slice(group)
        W_group = W[:, cols].float()
        act_group = act[cols].float()
        row, local_col = top_abs_candidate(W_group)
        v0.append({"group": group, "row": row, "local_col": local_col})

        row, local_col = candidate_rule_a(W_group, act_group)
        selected["ruleA"].append({"group": group, "row": row, "local_col": local_col})

        local_dead = [col - cols.start for col in range(cols.start, cols.stop) if col in dead_cols]
        cand_b = candidate_rule_b(W_group, torch.tensor(local_dead, dtype=torch.long))
        if cand_b is not None:
            selected["ruleB"].append({"group": group, "row": cand_b[0], "local_col": cand_b[1]})

        local_below = torch.nonzero(act_group < act_median, as_tuple=False).flatten()
        cand_c = candidate_rule_c(W_group, local_below)
        if cand_c is not None:
            selected["ruleC"].append({"group": group, "row": cand_c[0], "local_col": cand_c[1]})

    return selected, v0


def evaluate_candidates(
    W: torch.Tensor,
    X: torch.Tensor,
    layer_name: str,
    candidates: list[dict[str, int]],
    rule: str,
) -> list[dict[str, Any]]:
    records = []
    for item in candidates:
        record = evaluate_candidate(
            W, X, layer_name, item["group"], item["row"], item["local_col"],
            rule=rule, group_size=GROUP_SIZE, n_bits=N_BITS, zero_point=True, eta=ETA,
        )
        record["n_activation_tokens"] = int(X.shape[0])
        records.append(record)
    return records


def summarize_records(layer_name: str, rule: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "layer": layer_name, "rule": rule, "n_groups": 0,
            "n_B_positive": 0, "n_B_positive_and_A_below_median": 0,
            "max_B": float("nan"), "max_score": float("nan"),
        }
    A_values = torch.tensor([row["A_local_g"] for row in records], dtype=torch.float32)
    median_A = float(torch.median(A_values).item())
    return {
        "layer": layer_name,
        "rule": rule,
        "n_groups": len(records),
        "n_B_positive": sum(row["B_g"] > 0 for row in records),
        "n_B_positive_and_A_below_median": sum(row["B_g"] > 0 and row["A_local_g"] < median_A for row in records),
        "max_B": max(row["B_g"] for row in records),
        "max_score": max(row["score_g"] for row in records),
    }


def print_rule_summary(summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    print(f"\n{summary['layer']} {summary['rule']}")
    print(
        f"B>0: {summary['n_B_positive']} / {summary['n_groups']} | "
        f"B>0 and low A: {summary['n_B_positive_and_A_below_median']} | "
        f"max B: {summary['max_B']:.6e} | max score: {summary['max_score']:.6e}"
    )
    print("top-5 by score: group row col |w| B A score")
    for row in sorted(records, key=lambda item: item["score_g"], reverse=True)[:5]:
        print(
            f"{row['group']:2d} {row['row']:4d} {row['col']:5d} "
            f"{row['weight_magnitude']:.4e} {row['B_g']:.4e} "
            f"{row['A_local_g']:.4e} {row['score_g']:.4e}"
        )


def axis_limits(all_records: list[list[dict[str, Any]]]) -> tuple[tuple[float, float], tuple[float, float]]:
    flat = [row for records in all_records for row in records]
    xs = [max(row["A_local_g"], 1e-10) for row in flat] or [1e-10, 1.0]
    ys = [row["B_g"] for row in flat] or [-1.0, 1.0]
    y_min, y_max = min(ys), max(ys)
    pad = 0.05 * max(y_max - y_min, 1e-9)
    return (min(xs) * 0.8, max(xs) * 1.25), (y_min - pad, y_max + pad)


def plot_panel(ax: Any, records: list[dict[str, Any]], baseline: list[dict[str, Any]], title: str) -> None:
    # explanation: the gray v0 points are a methodological baseline; they show
    # whether changing the candidate rule actually moved dots toward upper-left.
    ax.scatter(
        [max(row["A_local_g"], 1e-10) for row in baseline],
        [row["B_g"] for row in baseline],
        c="gray", alpha=0.25, edgecolors="none", label="v0 top |w|",
    )
    top = sorted(records, key=lambda row: row["score_g"], reverse=True)[:10]
    top_keys = {(row["group"], row["row"], row["col"]) for row in top}
    colors = ["red" if (row["group"], row["row"], row["col"]) in top_keys else "steelblue" for row in records]
    ax.scatter(
        [max(row["A_local_g"], 1e-10) for row in records],
        [row["B_g"] for row in records],
        c=colors, alpha=0.85, edgecolors="none",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)


def save_layer_plot(
    layer_name: str,
    rule_records: dict[str, list[dict[str, Any]]],
    baseline: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xlim, ylim = axis_limits(list(rule_records.values()) + [baseline])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    for ax, rule in zip(axes, RULES):
        plot_panel(ax, rule_records[rule], baseline, rule)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    axes[0].set_ylabel("Local quantization benefit B (higher is better)")
    for ax in axes:
        ax.set_xlabel("FP16 layer change A_local (lower is safer)")
    fig.suptitle(f"RFIC v1 candidate rules: {layer_name}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_combined_plot(
    all_results: dict[str, dict[str, list[dict[str, Any]]]],
    all_baselines: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_record_lists = []
    for layer_name, rule_records in all_results.items():
        all_record_lists.extend(rule_records.values())
        all_record_lists.append(all_baselines[layer_name])
    xlim, ylim = axis_limits(all_record_lists)
    layers = list(all_results.keys())
    fig, axes = plt.subplots(len(layers), len(RULES), figsize=(15, 4 * len(layers)), sharex=True, sharey=True)
    for i, layer_name in enumerate(layers):
        for j, rule in enumerate(RULES):
            ax = axes[i][j] if len(layers) > 1 else axes[j]
            plot_panel(ax, all_results[layer_name][rule], all_baselines[layer_name], f"{layer_slug(layer_name)} {rule}")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if j == 0:
                ax.set_ylabel("B benefit")
            if i == len(layers) - 1:
                ax.set_xlabel("A_local damage")
    fig.suptitle("RFIC v1: rows are layers, columns are candidate rules")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["layer", "rule", "n_groups", "n_B_positive", "n_B_positive_and_A_below_median", "max_B", "max_score"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    settings = resolve_settings(args, load_config(args.config))
    model, tokenizer = load_model(settings["model_name"])
    texts = flatten_strata(load_strata())
    layers = list(settings["layers"])

    buffers = collect_linear_inputs(
        model, tokenizer, texts, layer_names=layers, batch_size=1,
        max_length=settings["max_length"], max_tokens_per_layer=settings["max_tokens"], device="cuda",
    )
    output_dir = Path(settings["output_dir"])
    figure_dir = Path(settings["figure_dir"])
    all_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    all_baselines: dict[str, list[dict[str, Any]]] = {}
    summary_rows = []

    for layer_name in layers:
        W = model.get_submodule(layer_name).weight.detach().float().cpu()
        X = buffers[layer_name].detach().float().cpu()
        assert W.ndim == 2 and X.ndim == 2 and X.shape[1] == W.shape[1]
        selected, v0_candidates = select_candidates_for_layer(W, X)
        baseline = evaluate_candidates(W, X, layer_name, v0_candidates, "v0_top_abs")
        rule_records = {rule: evaluate_candidates(W, X, layer_name, selected[rule], rule) for rule in RULES}
        all_baselines[layer_name] = baseline
        all_results[layer_name] = rule_records

        slug = layer_slug(layer_name)
        for rule in RULES:
            json_path = output_dir / f"uo_v1_{slug}_{rule}.json"
            write_json(json_path, rule_records[rule])
            summary = summarize_records(layer_name, rule, rule_records[rule])
            print_rule_summary(summary, rule_records[rule])
            summary_rows.append(summary)
        save_layer_plot(layer_name, rule_records, baseline, figure_dir / f"uo_v1_{slug}_3rules.png")

    save_combined_plot(all_results, all_baselines, Path(settings["combined_fig"]))
    write_summary_csv(Path(settings["summary_csv"]), summary_rows)
    print("\nlayer,rule,n_groups,n_B_positive,n_B_positive_and_A_below_median,max_B,max_score")
    for row in summary_rows:
        print(",".join(str(row[field]) for field in ["layer", "rule", "n_groups", "n_B_positive", "n_B_positive_and_A_below_median", "max_B", "max_score"]))
    print(f"\nsaved summary CSV: {settings['summary_csv']}")
    print(f"saved combined figure: {settings['combined_fig']}")


if __name__ == "__main__":
    main()
