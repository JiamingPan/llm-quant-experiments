"""
CLI entry point: depthwise scalar-mechanism diagnostic plots.

Usage:
    python scripts/run_diagnostics.py --config configs/qwen3_1b7.yaml
    python scripts/run_diagnostics.py --model-name /path/to/local/qwen3 \
        --uo-path results/uo_candidates.json \
        --output results/figures/depthwise_1b7.png

Pipeline:
    1. Load FP16 model
    2. Load a perturbed model copy
    3. If --uo-path given, load admitted UOs and apply zeroing to the perturbed copy
    4. Fixed test string (use a 200-token excerpt from WikiText-2 test set)
    5. Run compute_depthwise_diagnostics on all three variants:
         fp16, w2_baseline, w2_uo_zeroed
    6. Call plot_depthwise_comparison -> save to --output
    7. Print: mean bos_norm and sink_score for each variant at layers 0, n//2, n-1
"""

import argparse
import copy
import json
import torch
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Depthwise SW/UO interpretability diagnostics")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--uo-path", type=str, default=None,
                        help="JSON of admitted UO candidates from run_uo_detection.py")
    parser.add_argument("--output", type=str, default="results/figures/depthwise.png")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    model_cfg = cfg.get("model", {})
    quant_cfg = cfg.get("perturbation", {})
    eval_cfg = cfg.get("eval", {})
    output_cfg = cfg.get("output", {})

    model_name = model_cfg.get("name") or args.model_name
    if model_name is None:
        raise ValueError("Provide --config with model.name or pass --model-name")
    bits = quant_cfg.get("bits", args.bits)
    group_size = quant_cfg.get("group_size", args.group_size)
    dataset_name = eval_cfg.get("dataset", "Salesforce/wikitext")
    dataset_config = eval_cfg.get("dataset_config", "wikitext-2-raw-v1")
    dataset_split = eval_cfg.get("split", "test")
    output_path = output_cfg.get("diagnostics_plot", args.output)
    trust_remote_code = model_cfg.get("trust_remote_code", False)
    device_map = model_cfg.get("device_map", "auto")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from quant_analysis.diagnostics import compute_depthwise_diagnostics, plot_depthwise_comparison
    from quant_analysis.quantize import quantize_model_weights

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    quant_results = quantize_model_weights(model, bits=bits, group_size=group_size)
    model_q = copy.deepcopy(model)
    with torch.no_grad():
        for layer_name, (W_q, scales, zero_points, W_int) in quant_results.items():
            weight = model_q.get_submodule(layer_name).weight.data
            weight.copy_(W_q.to(device=weight.device, dtype=weight.dtype))
    model_q.eval()

    model_q_uo = copy.deepcopy(model_q)
    if args.uo_path:
        with open(args.uo_path) as f:
            admitted_uos = json.load(f)
        with torch.no_grad():
            for item in admitted_uos:
                layer_name = item["layer"]
                row = item["row"]
                col = item["col"]
                weight = model_q_uo.get_submodule(layer_name).weight.data
                weight[row, col] = torch.as_tensor(0.0, device=weight.device, dtype=weight.dtype)
    model_q_uo.eval()

    ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
    raw_text = "\n\n".join(t for t in ds["text"] if t.strip())
    token_ids = tokenizer(raw_text, add_special_tokens=False)["input_ids"][:200]
    text = tokenizer.decode(token_ids)

    d1 = compute_depthwise_diagnostics(model, tokenizer, text, device=args.device)
    d2 = compute_depthwise_diagnostics(model_q, tokenizer, text, device=args.device)
    d3 = compute_depthwise_diagnostics(model_q_uo, tokenizer, text, device=args.device)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plot_depthwise_comparison(
        {"fp16": d1, "w2_baseline": d2, "w2_uo": d3},
        labels=["FP16", f"W{bits} baseline", f"W{bits} + UO zeroed"],
        save_path=output_path,
    )

    for name, diagnostics in [("fp16", d1), ("w2_baseline", d2), ("w2_uo", d3)]:
        n_layers = diagnostics["n_layers"]
        layer_indices = [0, n_layers // 2, n_layers - 1]
        mean_bos_norm = sum(diagnostics["bos_norm"][idx] for idx in layer_indices) / len(layer_indices)
        mean_sink_score = sum(diagnostics["sink_score"][idx] for idx in layer_indices) / len(layer_indices)
        print(
            f"{name}: layers {layer_indices} | "
            f"mean bos_norm={mean_bos_norm:.6f} | mean sink_score={mean_sink_score:.6f}"
        )


if __name__ == "__main__":
    main()
