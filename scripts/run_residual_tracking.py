#!/usr/bin/env python
"""
Track residual-stream perturbations for selected scalar coordinates.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weight_handles.residual import track_residual_perturbation
from weight_handles.strata import flatten_strata, load_strata


def parse_args():
    parser = argparse.ArgumentParser(description="Residual-stream tracking for scalar handles")
    parser.add_argument("--config", default="configs/qwen3_1b7.yaml")
    parser.add_argument("--candidates", default="results/feature_handles.json")
    parser.add_argument("--output", default="results/residual_tracking.json")
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_model_and_tokenizer(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model_name = cfg.get("model", {}).get("name")
    if not model_name:
        raise ValueError("Config must contain model.name")

    with open(args.candidates, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    candidates = payload.get("candidates", payload)
    candidates = sorted(candidates, key=lambda item: item.get("ablation", {}).get("mean_kl", 0.0), reverse=True)
    candidates = candidates[:args.max_candidates]

    model, tokenizer = load_model_and_tokenizer(model_name)
    texts = flatten_strata(load_strata())

    results = []
    for candidate in tqdm(candidates, desc="residual"):
        coordinate = (candidate["layer"], int(candidate["row"]), int(candidate["col"]))
        tracking = track_residual_perturbation(
            model,
            tokenizer,
            texts,
            [coordinate],
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
        results.append({
            "candidate": candidate,
            "residual": tracking,
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"model": model_name, "results": results}, handle, indent=2)
    print(f"Saved residual tracking to {output_path}")


if __name__ == "__main__":
    main()
