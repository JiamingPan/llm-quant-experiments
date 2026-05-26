#!/usr/bin/env python
"""
Full-model RFIC gate for the top v1 Under-Outlier candidates.
It reads local RFIC candidates, temporarily zeroes each candidate before
4-bit groupwise quantizing the tested layers, and measures whether full-model
KL/PPL improves while the original FP16 model barely changes.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awq.core import pseudo_quantize_tensor
from weight_handles.metrics import (
    empty_comparison_stats,
    finalize_comparison_stats,
    logits_from_output,
    model_device,
    tokenized_batches,
    update_comparison_stats,
)
from weight_handles.strata import flatten_strata, load_strata
from weight_handles.uo import layer_slug, temporary_weight_replacements


MODEL_PATH = "/scratch/huterer_root/huterer0/jiamingp/models/qwen3-1b7"
DEFAULT_LAYERS = [
    "model.layers.7.mlp.down_proj",
    "model.layers.14.mlp.down_proj",
    "model.layers.20.mlp.down_proj",
]
DEFAULT_RULES = ["ruleB", "ruleC"]
N_BITS = 4
GROUP_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-model RFIC gate for top v1 candidates")
    parser.add_argument("--config", default="configs/uo_v1_gate.yaml")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--layers", nargs="*", default=None)
    parser.add_argument("--rules", nargs="*", default=None)
    parser.add_argument("--top-k-per-layer-rule", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-fig", default=None)
    parser.add_argument("--tail-fraction", type=float, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_settings(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    gate_cfg = cfg.get("rfic_v1_gate", {})
    return {
        "model_name": args.model_name or cfg.get("model", {}).get("name", MODEL_PATH),
        "results_dir": args.results_dir or gate_cfg.get("results_dir", "results"),
        "layers": args.layers or gate_cfg.get("layers", DEFAULT_LAYERS),
        "rules": args.rules or gate_cfg.get("rules", DEFAULT_RULES),
        "top_k_per_layer_rule": args.top_k_per_layer_rule or gate_cfg.get("top_k_per_layer_rule", 5),
        "max_length": args.max_length or gate_cfg.get("max_length", 512),
        "batch_size": args.batch_size or gate_cfg.get("batch_size", 1),
        "output_json": args.output_json or gate_cfg.get("output_json", "results/uo_v1_gate.json"),
        "output_csv": args.output_csv or gate_cfg.get("output_csv", "results/uo_v1_gate.csv"),
        "output_fig": args.output_fig or gate_cfg.get("output_fig", "results/figures/uo_v1_gate.png"),
        "tail_fraction": args.tail_fraction or gate_cfg.get("tail_fraction", 0.05),
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


def load_top_candidates(
    results_dir: Path,
    layers: list[str],
    rules: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    candidates = []
    seen = set()
    for layer_name in layers:
        slug = layer_slug(layer_name)
        for rule in rules:
            path = results_dir / f"uo_v1_{slug}_{rule}.json"
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as handle:
                records = json.load(handle)
            # explanation: the full-model gate is expensive, so we only test
            # candidates that already had positive local quantization benefit.
            good = [row for row in records if row["B_g"] > 0]
            good = sorted(good, key=lambda row: row["score_g"], reverse=True)[:top_k]
            for row in good:
                key = (row["layer"], int(row["row"]), int(row["col"]))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(row)
    return candidates


def evaluate_replacements(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    replacements: dict[str, torch.Tensor],
    batch_size: int,
    max_length: int,
    tail_fraction: float,
    device: str = "cuda",
) -> dict[str, float]:
    input_device = model_device(model, device)
    was_training = model.training
    model.eval()
    stats = empty_comparison_stats()

    try:
        with torch.no_grad():
            for input_ids, attention_mask in tokenized_batches(
                tokenizer, texts, batch_size=batch_size, max_length=max_length, device=input_device
            ):
                kwargs = {"input_ids": input_ids}
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask
                teacher_logits = logits_from_output(model(**kwargs))

                # explanation: teacher logits are always from the original FP16
                # model; only the student pass sees temporary zero/quantized weights.
                with temporary_weight_replacements(model, replacements):
                    student_logits = logits_from_output(model(**kwargs))
                update_comparison_stats(stats, teacher_logits, student_logits, input_ids, attention_mask)
    finally:
        model.train(was_training)

    return finalize_comparison_stats(stats, tail_fraction=tail_fraction)


def quantize_weight(W: torch.Tensor, n_bits: int = N_BITS, group_size: int = GROUP_SIZE) -> torch.Tensor:
    # explanation: "groupwise quantization" means each 128-column chunk gets
    # its own 4-bit rounding grid; this mirrors the local RFIC score definition.
    return pseudo_quantize_tensor(W, n_bits=n_bits, group_size=group_size, zero_point=True).detach()


def make_zeroed_quant_weight(original_weights: dict[str, torch.Tensor], candidate: dict[str, Any]) -> torch.Tensor:
    layer_name = candidate["layer"]
    W_zero = original_weights[layer_name].clone()
    # explanation: Z_P(W) is the RFIC intervention: set the candidate scalar to
    # zero before quantization and ask whether the resulting model is better.
    W_zero[int(candidate["row"]), int(candidate["col"])] = 0.0
    return quantize_weight(W_zero)


def make_zeroed_fp16_weight(original_weights: dict[str, torch.Tensor], candidate: dict[str, Any]) -> torch.Tensor:
    W_zero = original_weights[candidate["layer"]].clone()
    W_zero[int(candidate["row"]), int(candidate["col"])] = 0.0
    return W_zero


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer", "rule", "group", "row", "col", "B_g", "A_local_g", "score_g",
        "fp16_zero_mean_kl", "fp16_zero_delta_ppl", "fp16_zero_flip_rate",
        "q_baseline_mean_kl", "q_zero_mean_kl", "q_kl_improvement",
        "q_baseline_student_ppl", "q_zero_student_ppl", "q_ppl_improvement",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def save_gate_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [max(row["fp16_zero_mean_kl"], 1e-12) for row in rows]
    ys = [row["q_kl_improvement"] for row in rows]
    colors = ["red" if row["q_kl_improvement"] > 0 else "steelblue" for row in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, c=colors, alpha=0.85, edgecolors="none")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Full-model FP16 damage from zeroing (mean KL, lower is safer)")
    ax.set_ylabel("Full-model quantized KL improvement (higher is better)")
    ax.set_title("RFIC v1 gate: red points improve quantized full-model KL")
    ax.grid(True, which="both", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    settings = resolve_settings(args, load_config(args.config))
    candidates = load_top_candidates(
        Path(settings["results_dir"]),
        list(settings["layers"]),
        list(settings["rules"]),
        int(settings["top_k_per_layer_rule"]),
    )
    if not candidates:
        raise RuntimeError("No positive-B candidates found. Run scripts/run_uo_v1.py first.")

    model, tokenizer = load_model(settings["model_name"])
    texts = flatten_strata(load_strata())
    candidate_layers = sorted({row["layer"] for row in candidates})

    original_weights = {
        layer_name: model.get_submodule(layer_name).weight.detach().clone()
        for layer_name in candidate_layers
    }
    # explanation: this is not a production quantized model; it is a controlled
    # full-model forward test where only the RFIC-tested layers are quantized.
    q_baseline_weights = {
        layer_name: quantize_weight(weight)
        for layer_name, weight in original_weights.items()
    }
    q_baseline_metrics = evaluate_replacements(
        model, tokenizer, texts, q_baseline_weights,
        batch_size=int(settings["batch_size"]),
        max_length=int(settings["max_length"]),
        tail_fraction=float(settings["tail_fraction"]),
    )

    rows = []
    print(
        "layer rule group row col B A local_score "
        "fp16_KL q_KL_improvement q_PPL_improvement"
    )
    for candidate in candidates:
        layer_name = candidate["layer"]

        fp16_zero_metrics = evaluate_replacements(
            model,
            tokenizer,
            texts,
            {layer_name: make_zeroed_fp16_weight(original_weights, candidate)},
            batch_size=int(settings["batch_size"]),
            max_length=int(settings["max_length"]),
            tail_fraction=float(settings["tail_fraction"]),
        )

        q_zero_weights = dict(q_baseline_weights)
        q_zero_weights[layer_name] = make_zeroed_quant_weight(original_weights, candidate)
        q_zero_metrics = evaluate_replacements(
            model,
            tokenizer,
            texts,
            q_zero_weights,
            batch_size=int(settings["batch_size"]),
            max_length=int(settings["max_length"]),
            tail_fraction=float(settings["tail_fraction"]),
        )

        row = dict(candidate)
        row.update({
            "fp16_zero_mean_kl": fp16_zero_metrics["mean_kl"],
            "fp16_zero_delta_ppl": fp16_zero_metrics["delta_ppl"],
            "fp16_zero_flip_rate": fp16_zero_metrics["top_token_flip_rate"],
            "q_baseline_mean_kl": q_baseline_metrics["mean_kl"],
            "q_zero_mean_kl": q_zero_metrics["mean_kl"],
            "q_kl_improvement": q_baseline_metrics["mean_kl"] - q_zero_metrics["mean_kl"],
            "q_baseline_student_ppl": q_baseline_metrics["student_ppl"],
            "q_zero_student_ppl": q_zero_metrics["student_ppl"],
            "q_ppl_improvement": q_baseline_metrics["student_ppl"] - q_zero_metrics["student_ppl"],
        })
        rows.append(row)
        print(
            f"{row['layer']} {row['rule']} {row['group']} {row['row']} {row['col']} "
            f"{row['B_g']:.3e} {row['A_local_g']:.3e} {row['score_g']:.3e} "
            f"{row['fp16_zero_mean_kl']:.3e} {row['q_kl_improvement']:.3e} "
            f"{row['q_ppl_improvement']:.3e}"
        )

    rows = sorted(rows, key=lambda row: row["q_kl_improvement"], reverse=True)
    write_json(Path(settings["output_json"]), rows)
    write_csv(Path(settings["output_csv"]), rows)
    save_gate_plot(Path(settings["output_fig"]), rows)
    print(f"\nsaved gate JSON: {settings['output_json']}")
    print(f"saved gate CSV: {settings['output_csv']}")
    print(f"saved gate figure: {settings['output_fig']}")


if __name__ == "__main__":
    main()
