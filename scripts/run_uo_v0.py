#!/usr/bin/env python
"""
RFIC v0 tests one idea on one transformer layer: can zeroing one extreme
weight per quantization group reduce local low-bit error while barely changing
the original FP16 layer output? The scatter plot shows FP16 damage on x and
quantization benefit on y; the hoped-for signal is dots in the upper-left.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awq.core import collect_linear_inputs
from weight_handles.strata import flatten_strata, load_strata
from weight_handles.uo import evaluate_candidate, top_abs_candidate


MODEL_PATH = "/scratch/huterer_root/huterer0/jiamingp/models/qwen3-1b7"
DEFAULT_LAYER = "model.layers.14.mlp.down_proj"
# explanation: 4-bit quantization stores each weight using 16 possible values,
# so it is a strong compression stress test but still common in AWQ-style work.
N_BITS = 4
# explanation: group size is the number of neighboring input columns that share
# one quantization grid; RFIC asks if one extreme value hurts that shared grid.
GROUP_SIZE = 128
ETA = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RFIC v0 on one Qwen3 MLP layer")
    parser.add_argument("--layer-name", default=DEFAULT_LAYER)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output-json", default="results/uo_v0_layer14_downproj.json")
    parser.add_argument("--output-fig", default="results/figures/uo_v0_scatter_layer14_downproj.png")
    parser.add_argument("--model-name", default=MODEL_PATH)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


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


def evaluate_group(
    W: torch.Tensor,
    X: torch.Tensor,
    group_index: int,
    group_size: int = GROUP_SIZE,
) -> dict[str, Any]:
    start = group_index * group_size
    end = min(start + group_size, W.shape[1])
    W_group = W[:, start:end].detach().float()
    assert X[:, start:end].shape[1] == W_group.shape[1]
    assert W_group.shape[1] <= group_size

    # explanation: groupwise quantization means each contiguous block of input
    # columns gets its own low-bit grid; group_size controls how many columns
    # share that grid, so one extreme value can affect many neighboring weights.
    row, local_col = top_abs_candidate(W_group)
    # explanation: v0 deliberately tests the simplest packet rule: zero the
    # single largest-magnitude scalar in each group and measure B versus A.
    return evaluate_candidate(
        W, X, DEFAULT_LAYER, group_index, row, local_col, rule="v0_top_abs",
        group_size=group_size, n_bits=N_BITS, zero_point=True, eta=ETA,
    )


def save_scatter(records: list[dict[str, Any]], output_fig: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top10 = sorted(records, key=lambda item: item["score_g"], reverse=True)[:10]
    top10_keys = {(item["group"], item["row"], item["col"]) for item in top10}

    xs = [max(item["A_local_g"], 1e-10) for item in records]
    ys = [item["B_g"] for item in records]
    colors = ["red" if (item["group"], item["row"], item["col"]) in top10_keys else "steelblue" for item in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, c=colors, alpha=0.85, edgecolors="none")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("FP16 layer change after zeroing candidate (A_local, lower is safer)")
    ax.set_ylabel("Local quantization benefit from zeroing candidate (B, higher is better)")
    ax.set_title("RFIC v0: upper-left dots are useful Under-Outlier candidates")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    output_path = Path(output_fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_summary(records: list[dict[str, Any]]) -> None:
    positives = [item for item in records if item["B_g"] > 0]
    A_values = torch.tensor([item["A_local_g"] for item in records], dtype=torch.float32)
    median_A = float(torch.median(A_values).item())
    safe_positives = [item for item in records if item["B_g"] > 0 and item["A_local_g"] < median_A]
    B_values = torch.tensor([item["B_g"] for item in records], dtype=torch.float32)
    top10 = sorted(records, key=lambda item: item["score_g"], reverse=True)[:10]

    print("\nRFIC v0 summary")
    print(f"groups evaluated: {len(records)}")
    print(f"groups with B_g > 0: {len(positives)}")
    print(f"groups with B_g > 0 and A_local_g < median(A): {len(safe_positives)}")
    print(f"mean B_g: {float(B_values.mean().item()):.6e}")
    print(f"max B_g: {float(B_values.max().item()):.6e}")
    print("\ntop-10 packets by score")
    print("group | row | col | |w| | B_g | A_local_g | score_g")
    for item in top10:
        print(
            f"{item['group']:5d} | {item['row']:4d} | {item['col']:5d} | "
            f"{item['weight_magnitude']:.6e} | {item['B_g']:.6e} | "
            f"{item['A_local_g']:.6e} | {item['score_g']:.6e}"
        )


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model(args.model_name)
    layer = model.get_submodule(args.layer_name)
    W = layer.weight.detach().float().cpu()
    assert W.ndim == 2
    assert W.shape[1] % GROUP_SIZE == 0

    texts = flatten_strata(load_strata())
    # explanation: these are real inputs to the chosen linear layer; local layer
    # error is data-dependent, so RFIC needs actual activation vectors X.
    buffers = collect_linear_inputs(
        model, tokenizer, texts, layer_names=[args.layer_name], batch_size=1,
        max_length=args.max_length, max_tokens_per_layer=args.max_tokens, device="cuda",
    )
    X = buffers[args.layer_name].detach().float().cpu()
    assert X.ndim == 2
    assert X.shape[1] == W.shape[1]

    n_groups = W.shape[1] // GROUP_SIZE
    records = [evaluate_group(W, X, group_index) for group_index in range(n_groups)]
    for item in records:
        item["layer"] = args.layer_name
        item["n_activation_tokens"] = int(X.shape[0])

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    save_scatter(records, args.output_fig)
    print_summary(records)
    print(f"\nsaved JSON: {output_json}")
    print(f"saved scatter: {args.output_fig}")


if __name__ == "__main__":
    main()
