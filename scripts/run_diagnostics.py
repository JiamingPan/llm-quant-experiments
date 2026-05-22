"""
CLI entry point: depthwise diagnostic plots.

Usage:
    python scripts/run_diagnostics.py --config configs/qwen3_1b7.yaml
    python scripts/run_diagnostics.py --model-name Qwen/Qwen3-1.7B \
        --uo-path results/uo_candidates.json \
        --output results/figures/depthwise_1b7.png

TODO (Codex): implement main() using the spec below.

Pipeline:
    1. Load FP16 model
    2. Load quantized model (apply group quantization)
    3. If --uo-path given, load admitted UOs and apply zeroing to quantized model
    4. Fixed test string (use a 200-token excerpt from WikiText-2 test set)
    5. Run compute_depthwise_diagnostics on all three variants:
         fp16, w2_baseline, w2_uo_zeroed
    6. Call plot_depthwise_comparison -> save to --output
    7. Print: mean bos_norm and sink_score for each variant at layers 0, n//2, n-1
"""

import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Depthwise diagnostics for quantized LLMs")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
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
            cfg = yaml.safe_load(f)

    # TODO (Codex): implement the full pipeline described in the docstring above
    raise NotImplementedError("TODO: implement main() in Codex")


if __name__ == "__main__":
    main()
