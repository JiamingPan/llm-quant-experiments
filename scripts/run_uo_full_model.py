#!/usr/bin/env python
"""
Run the full-model RFIC check for v1 Under-Outlier candidates.

The four-variant comparison isolates cause: FP16 is the reference, AWQ is the
stock W4G128 quantized model, UO-then-AWQ tests whether masking adds value, and
UO-only-FP16 tests whether the mask itself damages the original model.
Bootstrap stderr turns a PPL number into a measurement; tiny changes can be
sampling noise. WikiText-2 validation is the standard quantization eval corpus.
The v1 median-A_local filter keeps only candidates that passed the local safety
condition before we spend time on full-model testing.
"""

import argparse
import csv
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awq.core import collect_linear_inputs, temporary_awq
from weight_handles.metrics import lm_loss_from_logits, logits_from_output, model_device
from weight_handles.strata import flatten_strata, load_strata


MODEL_PATH = "/scratch/huterer_root/huterer0/jiamingp/models/qwen3-1b7"
TARGET_SUFFIXES = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
N_BITS = 4
GROUP_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-model RFIC UO PPL check")
    parser.add_argument("--model-name", default=MODEL_PATH)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--n-eval-chunks", type=int, default=256)
    parser.add_argument("--chunk-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ruleset", default="B", choices=["A", "B", "C", "all"])
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--calib-max-length", type=int, default=256)
    parser.add_argument("--calib-max-tokens-per-layer", type=int, default=1024)
    parser.add_argument("--output-table", default="results/uo_full_model_table.csv")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-fig", default=None)
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


def load_wikitext_split(split: str):
    from datasets import load_dataset

    try:
        return load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    except Exception:
        # explanation: some newer Hugging Face/datasets installs require the
        # namespace-qualified dataset id even though the corpus is the same.
        return load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)


def make_eval_chunks(tokenizer: Any, n_chunks: int, chunk_len: int) -> list[torch.Tensor]:
    ds = load_wikitext_split("validation")
    text = "\n\n".join(item for item in ds["text"] if item.strip())
    # explanation: we tokenize once, then slice fixed non-overlapping windows so
    # every model variant is evaluated on exactly the same token chunks.
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
    usable = min(int(ids.numel() // chunk_len), n_chunks)
    chunks = [ids[i * chunk_len:(i + 1) * chunk_len].unsqueeze(0).contiguous() for i in range(usable)]
    assert all(chunk.shape == (1, chunk_len) for chunk in chunks)
    return chunks


def calibration_texts() -> list[str]:
    strata_texts = flatten_strata(load_strata())[:8]
    ds = load_wikitext_split("train")
    wiki_texts = [item for item in ds["text"] if len(item.strip()) > 50][:8]
    # explanation: AWQ scale search needs real activations; mixing small strata
    # prompts and WikiText train text gives a deterministic calibration set.
    return strata_texts + wiki_texts


def target_linear_names(model: Any) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(TARGET_SUFFIXES):
            names.append(name)
    return names


def selected_rules(ruleset: str) -> set[str]:
    if ruleset == "all":
        return {"ruleA", "ruleB", "ruleC"}
    return {f"rule{ruleset}"}


def load_uo_mask(results_dir: Path, ruleset: str, k_per_layer: int) -> tuple[dict[str, list[tuple[int, int]]], dict[str, dict[str, int]], list[dict[str, Any]]]:
    rules = selected_rules(ruleset)
    by_layer: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, dict[str, int]] = {}

    for path in sorted(results_dir.glob("uo_v1_*_rule*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not records or records[0].get("rule") not in rules:
            continue
        median_a = torch.median(torch.tensor([row["A_local_g"] for row in records], dtype=torch.float32)).item()
        # explanation: RFIC has two conditions: useful for quantization (B>0)
        # and locally safe in FP16 (A below that file's median damage).
        kept = [row for row in records if row["B_g"] > 0 and row["A_local_g"] < median_a]
        for row in kept:
            by_layer.setdefault(row["layer"], []).append(row)

    selected = []
    seen = set()
    for layer_name, rows in by_layer.items():
        rows = sorted(rows, key=lambda row: row["score_g"], reverse=True)
        layer_take = []
        for row in rows:
            key = (row["layer"], int(row["row"]), int(row["col"]))
            if key in seen:
                continue
            seen.add(key)
            layer_take.append(row)
            if len(layer_take) >= k_per_layer:
                break
        selected.extend(layer_take)

    uo_mask: dict[str, list[tuple[int, int]]] = {}
    for row in selected:
        layer_name = row["layer"]
        rule = row["rule"]
        uo_mask.setdefault(layer_name, []).append((int(row["row"]), int(row["col"])))
        counts.setdefault(layer_name, {}).setdefault(rule, 0)
        counts[layer_name][rule] += 1

    return uo_mask, counts, selected


@contextmanager
def temporary_zero_uo(model: Any, uo_mask: dict[str, list[tuple[int, int]]]):
    originals = []
    try:
        with torch.no_grad():
            for layer_name, coords in uo_mask.items():
                weight = model.get_submodule(layer_name).weight
                for row, col in coords:
                    originals.append((weight, row, col, weight.data[row, col].detach().clone()))
                    # explanation: Z_S(W) means "zero the candidate UO set S"
                    # before quantization, so a harmful grid-stretching scalar is removed.
                    weight.data[row, col] = 0.0
        yield
    finally:
        with torch.no_grad():
            for weight, row, col, original in reversed(originals):
                weight.data[row, col].copy_(original)


def chunk_losses(model: Any, chunks: list[torch.Tensor], desc: str) -> list[dict[str, float]]:
    device = model_device(model, "cuda")
    was_training = model.training
    model.eval()
    rows = []
    try:
        with torch.no_grad():
            for chunk in tqdm(chunks, desc=desc, unit="chunk"):
                input_ids = chunk.to(device)
                assert input_ids.ndim == 2 and input_ids.shape[0] == 1
                logits = logits_from_output(model(input_ids=input_ids))
                nll, n_tokens = lm_loss_from_logits(logits, input_ids)
                rows.append({
                    "nll": float(nll),
                    "n_tokens": int(n_tokens),
                    "ppl": float(math.exp(nll / max(n_tokens, 1))),
                })
    finally:
        model.train(was_training)
    return rows


def bootstrap_mean_stderr(values: torch.Tensor, seed: int, n_resamples: int) -> tuple[float, float]:
    assert values.ndim == 1
    mean = float(values.mean().item())
    if values.numel() < 2:
        return mean, 0.0
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randint(values.numel(), (n_resamples, values.numel()), generator=generator)
    boot_means = values[idx].mean(dim=1)
    return mean, float(boot_means.std(unbiased=True).item())


def summarize_ppl(losses: list[dict[str, float]], seed: int, n_resamples: int) -> dict[str, Any]:
    ppls = torch.tensor([row["ppl"] for row in losses], dtype=torch.float64)
    mean, stderr = bootstrap_mean_stderr(ppls, seed, n_resamples)
    total_nll = sum(row["nll"] for row in losses)
    total_tokens = sum(row["n_tokens"] for row in losses)
    return {
        "ppl": mean,
        "stderr": stderr,
        "global_ppl": float(math.exp(total_nll / max(total_tokens, 1))),
        "n_chunks": len(losses),
        "n_tokens": int(total_tokens),
    }


def measure_variant(model: Any, chunks: list[torch.Tensor], name: str, seed: int, n_resamples: int) -> dict[str, Any]:
    losses = chunk_losses(model, chunks, name)
    result = summarize_ppl(losses, seed, n_resamples)
    result["variant"] = name
    return result


def table_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    fp16 = results["fp16"]["ppl"]
    awq = results["awq_baseline"]["ppl"]
    rows = []
    for name in ["fp16", "awq_baseline", "uo_then_awq", "uo_only_fp16"]:
        ppl = results[name]["ppl"]
        rows.append({
            "variant": name,
            "ppl": ppl,
            "stderr": results[name]["stderr"],
            "delta_vs_fp16": ppl - fp16,
            "delta_vs_awq_baseline": "" if name in {"fp16", "uo_only_fp16"} else ppl - awq,
            "global_ppl": results[name]["global_ppl"],
        })
    return rows


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "ppl", "stderr", "delta_vs_fp16", "delta_vs_awq_baseline", "global_ppl"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verdict(results: dict[str, dict[str, Any]]) -> str:
    fp16 = results["fp16"]
    awq = results["awq_baseline"]
    uo_awq = results["uo_then_awq"]
    uo_fp16 = results["uo_only_fp16"]
    gain = awq["ppl"] - uo_awq["ppl"]
    if uo_awq["ppl"] + uo_awq["stderr"] < awq["ppl"] - awq["stderr"]:
        return f"RFIC: UO masking improves quantized PPL by {gain:.4f} (>2σ)."
    awq_same = abs(uo_awq["ppl"] - awq["ppl"]) <= (uo_awq["stderr"] + awq["stderr"])
    fp16_same = abs(uo_fp16["ppl"] - fp16["ppl"]) <= (uo_fp16["stderr"] + fp16["stderr"])
    if awq_same and fp16_same:
        return "RFIC: UO masking is safe (no FP16 damage, no quantized improvement at K=top-K)."
    if not fp16_same:
        return "RFIC failure mode: UO mask harms FP16; A_local underestimated full-model damage."
    return "RFIC failure mode: UO masking changes quantized PPL but not as a statistically clear improvement."


def save_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["variant"] for row in rows]
    ppls = [float(row["ppl"]) for row in rows]
    errs = [float(row["stderr"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(labels, ppls, yerr=errs, fmt="o", capsize=4)
    span = max(ppls) - min(ppls)
    pad = max(span * 0.25, max(errs) * 3, 1e-3)
    # explanation: the y-axis is zoomed to the measured PPL regime; otherwise a
    # tiny but real UO-vs-AWQ difference can disappear visually.
    ax.set_ylim(min(ppls) - pad, max(ppls) + pad)
    ax.set_ylabel("WikiText-2 validation PPL (mean over chunks)")
    ax.set_title("Full-model RFIC PPL check with bootstrap stderr")
    ax.grid(True, axis="y", alpha=0.25)
    fig.autofmt_xdate(rotation=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_json = args.output_json or f"results/uo_full_model_K{args.K}.json"
    output_fig = args.output_fig or f"results/figures/uo_full_model_K{args.K}.png"

    uo_mask, counts, selected = load_uo_mask(Path(args.results_dir), args.ruleset, args.K)
    if not uo_mask:
        raise RuntimeError("No UO candidates selected. Run scripts/run_uo_v1.py first.")
    print("selected UO packets per layer/rule:")
    for layer_name, per_rule in counts.items():
        print(layer_name, per_rule)

    model, tokenizer = load_model(args.model_name)
    eval_chunks = make_eval_chunks(tokenizer, args.n_eval_chunks, args.chunk_len)
    print(f"eval chunks: {len(eval_chunks)} x {args.chunk_len} tokens")

    target_layers = target_linear_names(model)
    calib_texts = calibration_texts()
    print(f"collecting AWQ inputs for {len(target_layers)} linear layers")
    input_buffers = collect_linear_inputs(
        model, tokenizer, calib_texts, layer_names=target_layers, batch_size=1,
        max_length=args.calib_max_length, max_tokens_per_layer=args.calib_max_tokens_per_layer, device="cuda",
    )

    results = {}
    results["fp16"] = measure_variant(model, eval_chunks, "fp16", args.seed, args.bootstrap_resamples)
    with temporary_awq(
        model, input_buffers, layer_names=target_layers, n_bits=N_BITS,
        group_size=GROUP_SIZE, zero_point=True, progress=True,
    ):
        results["awq_baseline"] = measure_variant(model, eval_chunks, "awq_baseline", args.seed, args.bootstrap_resamples)
    with temporary_zero_uo(model, uo_mask):
        with temporary_awq(
            model, input_buffers, layer_names=target_layers, n_bits=N_BITS,
            group_size=GROUP_SIZE, zero_point=True, progress=True,
        ):
            results["uo_then_awq"] = measure_variant(model, eval_chunks, "uo_then_awq", args.seed, args.bootstrap_resamples)
    with temporary_zero_uo(model, uo_mask):
        results["uo_only_fp16"] = measure_variant(model, eval_chunks, "uo_only_fp16", args.seed, args.bootstrap_resamples)

    del input_buffers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows = table_rows(results)
    write_table(Path(args.output_table), rows)
    save_plot(Path(output_fig), rows)
    message = verdict(results)
    payload = {
        "model_name": args.model_name,
        "K": args.K,
        "ruleset": args.ruleset,
        "seed": args.seed,
        "uo_counts": counts,
        "selected_candidates": selected,
        "ppl": results,
        "eval_chunk_count": len(eval_chunks),
        "chunk_len": args.chunk_len,
        "calibration_source": "8 deterministic strata prompts + 8 WikiText-2 train samples",
        "verdict": message,
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("\nvariant,ppl,stderr,delta_vs_fp16,delta_vs_awq_baseline")
    for row in rows:
        print(f"{row['variant']},{row['ppl']:.6f},{row['stderr']:.6f},{row['delta_vs_fp16']:.6f},{row['delta_vs_awq_baseline']}")
    print(message)
    print(f"saved table: {args.output_table}")
    print(f"saved JSON: {output_json}")
    print(f"saved figure: {output_fig}")


if __name__ == "__main__":
    main()
