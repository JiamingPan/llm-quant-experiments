#!/usr/bin/env python
"""
Run the scalar weight-outlier measurement harness on a public language model.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weight_handles.ablation import evaluate_coordinate_ablation
from weight_handles.candidates import (
    dedupe_candidates,
    functional_under_outlier_candidates,
    known_super_weight_candidates,
    matched_control_candidates,
    top_magnitude_candidates,
)
from weight_handles.patching import patch_back_sufficiency, rank_one_patch_sufficiency
from weight_handles.residual import track_residual_perturbation
from weight_handles.strata import flatten_strata, load_strata
from weight_handles.taxonomy import classify_results


def parse_args():
    parser = argparse.ArgumentParser(description="Scalar weight-outlier measurement harness")
    parser.add_argument("--config", default="configs/qwen3_1b7.yaml")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output", default="results/feature_handles.json")
    parser.add_argument("--max-robust", type=int, default=64)
    parser.add_argument("--max-under", type=int, default=32)
    parser.add_argument("--max-controls", type=int, default=32)
    parser.add_argument("--max-residual", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def evaluate_by_stratum(model, tokenizer, candidate, strata, args):
    per_stratum = {}
    for name, texts in strata.items():
        per_stratum[name] = evaluate_coordinate_ablation(
            model,
            tokenizer,
            texts,
            candidate["layer"],
            int(candidate["row"]),
            int(candidate["col"]),
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
    return per_stratum


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    model_name = args.model_name or cfg.get("model", {}).get("name")
    if not model_name:
        raise ValueError("Model name must be supplied in config or --model-name")

    model, tokenizer = load_model_and_tokenizer(model_name)
    strata = load_strata()
    all_texts = flatten_strata(strata)

    known = known_super_weight_candidates(model, cfg.get("known_super_weights", []))
    robust = top_magnitude_candidates(model, k=args.max_robust)
    under = functional_under_outlier_candidates(
        model,
        tokenizer,
        all_texts,
        robust,
        max_candidates=args.max_under,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    controls = matched_control_candidates(
        model,
        dedupe_candidates(known + robust[:args.max_controls] + under),
        max_controls=args.max_controls,
    )
    candidates = dedupe_candidates(known + robust + under + controls)

    results = []
    for candidate in tqdm(candidates, desc="ablation"):
        item = dict(candidate)
        item["ablation"] = evaluate_coordinate_ablation(
            model,
            tokenizer,
            all_texts,
            candidate["layer"],
            int(candidate["row"]),
            int(candidate["col"]),
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
        item["strata"] = evaluate_by_stratum(model, tokenizer, candidate, strata, args)
        results.append(item)

    residual_targets = sorted(results, key=lambda item: item["ablation"]["mean_kl"], reverse=True)[:args.max_residual]
    for item in tqdm(residual_targets, desc="residual"):
        coordinate = (item["layer"], int(item["row"]), int(item["col"]))
        residual = track_residual_perturbation(
            model,
            tokenizer,
            all_texts,
            [coordinate],
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
        item["residual"] = {
            "layers": residual["layers"],
            "max_stability": max((layer["stability"] for layer in residual["layers"]), default=0.0),
        }
        item["patch_back"] = patch_back_sufficiency(
            model,
            tokenizer,
            all_texts,
            [coordinate],
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
        item["rank_one_patch"] = rank_one_patch_sufficiency(
            model,
            tokenizer,
            all_texts,
            [coordinate],
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )

    classified = classify_results(results)
    output = {
        "model": model_name,
        "seed": args.seed,
        "strata": {key: len(value) for key, value in strata.items()},
        "candidates": classified,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved {len(classified)} candidate results to {output_path}")


if __name__ == "__main__":
    main()
