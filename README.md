# llm-quant-experiments

Forward-pass quantization analysis for Qwen3 models. Detects Under-Outliers (UOs) and Super Weights (SWs), runs depthwise diagnostic curves, and evaluates 2-bit / 3-bit compression quality.

Built as part of LLM compression research at [Escha Labs](https://escha.ai).

## What this does

**UO detection** — identifies range-frontier weights: scalars at the max/min of their quantization group that inflate the group scale and hurt all other weights in the group. Admits them via a forward-pass gate: only zero a weight if it's both FP16-harmless (relative PPL damage < 0.5%) and quantization-useful (improves quantized model PPL).

**SW detection** — finds extreme outlier scalars in `mlp.down_proj` layers that are load-bearing for model quality. These are protected, not zeroed.

**Depthwise diagnostics** — compares FP16 vs W2-baseline vs W2+UO-zeroed on three per-layer signals: BOS token norm, attention sink score, and representation entropy. Checks whether UO zeroing recovers the FP16 sink/compression trajectory, not just PPL.

**Alpha sweep** — tests `W_sw → α * W_sw` for α ∈ {0.8, 0.9, 1.0, 1.1, 1.2} to find whether static FP16 restoration is optimal or whether the quantized downstream Jacobian prefers a different scale.

## Setup

```bash
git clone https://github.com/JiamingPan/llm-quant-experiments.git
cd llm-quant-experiments
pip install -r requirements.txt
```

## Usage

**UO detection (local or cluster):**
```bash
python scripts/run_uo_detection.py --config configs/qwen3_1b7.yaml
python scripts/run_uo_detection.py --config configs/qwen3_8b.yaml
```

**Depthwise diagnostics:**
```bash
python scripts/run_diagnostics.py \
    --config configs/qwen3_1b7.yaml \
    --uo-path results/uo_candidates_1b7.json \
    --output results/figures/depthwise_1b7.png
```

**Great Lakes (SLURM):**
```bash
# edit account and paths in the sbatch file first
sbatch scripts/slurm/uo_1b7.sbatch
sbatch scripts/slurm/uo_8b.sbatch
```

## Repository layout

```
quant_analysis/
    quantize.py      asymmetric per-group quantization (W2/W3, group_size=128)
    detect.py        range-frontier UO detection, SW detection
    metrics.py       delta-PPL, KL, EAR, UO admission gate
    diagnostics.py   depthwise BOS norm / sink score / rep entropy
scripts/
    run_uo_detection.py   main UO detection pipeline
    run_diagnostics.py    depthwise diagnostic plots
    slurm/               Great Lakes job scripts
configs/
    qwen3_1b7.yaml        Qwen3-1.7B config
    qwen3_8b.yaml         Qwen3-8B config
results/                 gitignored outputs
```

## Key concepts

**Group quantization** — weights are quantized in groups of 128 columns. The group scale = (max - min) / (2^b - 1). A range-frontier UO sits at the max or min, inflating the scale for all other weights.

**UO admission criterion** — both conditions must hold:
1. `delta_PPL_FP16 / baseline_PPL < 0.005` (FP16-harmless)
2. Quantized model PPL improves after zeroing (Q-useful)

**EAR** — Expected Acceptance Rate: fraction of tokens where FP16 and quantized model agree on argmax. EAR ≥ 0.99 = distribution-lossless (SLQ, Helcig et al. 2026).

## References

- Yu et al. 2024 — Super Weights ([arXiv:2411.07191](https://arxiv.org/abs/2411.07191))
- Lin et al. 2024 — AWQ ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978))
- Guo et al. 2025 — PQI/ReQuant ([arXiv:2503.01901](https://arxiv.org/abs/2503.01901))
- Helcig et al. 2026 — SLQ ([arXiv:2605.02404](https://arxiv.org/abs/2605.02404))
- Queipo-de-Llano et al. 2026 — Mix-Compress-Refine ([arXiv:2510.06477](https://arxiv.org/abs/2510.06477))
