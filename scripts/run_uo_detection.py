"""
CLI entry point: UO detection pipeline on Qwen3.

Usage:
    python scripts/run_uo_detection.py --config configs/qwen3_1b7.yaml
    python scripts/run_uo_detection.py --model-name Qwen/Qwen3-8B --bits 3

TODO (Codex): implement main() using the spec below.

Pipeline:
    1. Load model + tokenizer from config or --model-name flag
    2. Load calibration texts from WikiText-2 (datasets library)
       n_samples=128, max_length=512
    3. Compute baseline FP16 PPL  -> print
    4. Quantize all mlp layers at --bits with group_size=128
    5. Compute baseline quantized PPL -> print
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
import json
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="UO detection for quantized LLMs")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (overrides other flags if given)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
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
            cfg = yaml.safe_load(f)

    # TODO (Codex): implement the full pipeline described in the docstring above
    raise NotImplementedError("TODO: implement main() in Codex")


if __name__ == "__main__":
    main()
