"""
CLI entry point: UO scalar-mechanism detection pipeline on Qwen3.

Usage:
    python scripts/run_uo_detection.py --config configs/qwen3_1b7.yaml
    python scripts/run_uo_detection.py --model-name /path/to/local/qwen3 --bits 3

Pipeline:
    1. Load model + tokenizer from config or --model-name flag
    2. Load calibration texts from WikiText-2 (datasets library)
       n_samples=128, max_length=512
    3. Compute baseline FP16 PPL  -> print
    4. Apply the configured low-bit perturbation to model copies
    5. Compute perturbed-model PPL -> print
    6. For each mlp layer (down_proj, gate_proj, up_proj):
         a. find_range_frontier_candidates(W, group_size=128)
         b. For each candidate (packet_size=1):
              gate_uo_candidate(..., fp16_budget=0.005)
         c. Collect admitted UOs
    7. Save admitted UOs to --output as JSON
    8. Print summary table:
         Layer | Candidates | Admitted | Mean scale reduction | PPL delta
"""

import argparse
import copy
import json
import torch
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="UO scalar mechanism detection for Qwen3")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (overrides other flags if given)")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--fp16-budget", type=float, default=0.005)
    parser.add_argument("--output", type=str, default="results/uo_candidates.json")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config if given
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    model_cfg = cfg.get("model", {})
    quant_cfg = cfg.get("perturbation", {})
    uo_cfg = cfg.get("detection", {}).get("uo", {})
    calibration_cfg = cfg.get("calibration", {})
    output_cfg = cfg.get("output", {})

    model_name = model_cfg.get("name") or args.model_name
    if model_name is None:
        raise ValueError("Provide --config with model.name or pass --model-name")
    bits = quant_cfg.get("bits", args.bits)
    group_size = quant_cfg.get("group_size", args.group_size)
    dataset_name = calibration_cfg.get("dataset", "Salesforce/wikitext")
    dataset_config = calibration_cfg.get("dataset_config", "wikitext-2-raw-v1")
    dataset_split = calibration_cfg.get("split", "train")
    n_samples = calibration_cfg.get("n_samples", args.n_samples)
    fp16_budget = uo_cfg.get("fp16_ppl_budget", args.fp16_budget)
    max_candidates_per_layer = uo_cfg.get("max_candidates_per_layer", None)
    output_path = output_cfg.get("uo_candidates", args.output)
    trust_remote_code = model_cfg.get("trust_remote_code", False)
    device_map = model_cfg.get("device_map", "auto")

    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from quant_analysis.detect import find_range_frontier_candidates
    from quant_analysis.metrics import compute_perplexity, gate_uo_candidate
    from quant_analysis.quantize import quantize_model_weights

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:n_samples]

    baseline_fp16_ppl = compute_perplexity(model, tokenizer, texts)
    print(f"FP16 PPL: {baseline_fp16_ppl:.3f}")

    quant_results = quantize_model_weights(model, bits=bits, group_size=group_size)
    model_q = copy.deepcopy(model)
    with torch.no_grad():
        for layer_name, (W_q, scales, zero_points, W_int) in quant_results.items():
            weight = model_q.get_submodule(layer_name).weight.data
            weight.copy_(W_q.to(device=weight.device, dtype=weight.dtype))
    model_q.eval()

    baseline_q_ppl = compute_perplexity(model_q, tokenizer, texts)
    print(f"W{bits} PPL: {baseline_q_ppl:.3f}")

    admitted_uos = []
    summary_rows = []
    mlp_layers = [name for name in quant_results if ".mlp." in name]

    marker = object()
    old_model_q = getattr(model, "_quant_analysis_model_q", marker)
    try:
        setattr(model, "_quant_analysis_model_q", model_q)
        for layer_name in tqdm(mlp_layers, desc="Layers"):
            W_orig = model.get_submodule(layer_name).weight.data
            candidates = find_range_frontier_candidates(W_orig, group_size=group_size)
            if max_candidates_per_layer is not None:
                candidates = candidates[:max_candidates_per_layer]

            layer_admitted = []
            for candidate in tqdm(candidates, desc=layer_name, leave=False):
                result = gate_uo_candidate(
                    model,
                    tokenizer,
                    texts,
                    layer_name,
                    candidate["row"],
                    candidate["col"],
                    baseline_fp16_ppl,
                    baseline_q_ppl,
                    fp16_budget=fp16_budget,
                )
                if result["admitted"]:
                    entry = {"layer": layer_name}
                    entry.update(candidate)
                    entry.update(result)
                    admitted_uos.append(entry)
                    layer_admitted.append(entry)

            mean_scale_reduction = (
                sum(item["scale_reduction"] for item in layer_admitted) / len(layer_admitted)
                if layer_admitted else 0.0
            )
            mean_q_improvement = (
                sum(item["q_improvement"] for item in layer_admitted) / len(layer_admitted)
                if layer_admitted else 0.0
            )
            summary_rows.append((layer_name, len(candidates), len(layer_admitted),
                                 mean_scale_reduction, mean_q_improvement))
    finally:
        if old_model_q is marker:
            delattr(model, "_quant_analysis_model_q")
        else:
            setattr(model, "_quant_analysis_model_q", old_model_q)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(admitted_uos, f, indent=2)

    print("Layer | Candidates | Admitted | Mean scale reduction | PPL delta")
    for layer_name, n_candidates, n_admitted, mean_scale_reduction, mean_q_improvement in summary_rows:
        print(
            f"{layer_name} | {n_candidates} | {n_admitted} | "
            f"{mean_scale_reduction:.6f} | {mean_q_improvement:.6f}"
        )
    print(f"Saved {len(admitted_uos)} admitted UOs to {output_path}")


if __name__ == "__main__":
    main()
