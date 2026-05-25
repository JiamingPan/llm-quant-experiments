# llm-quant-experiments

Private early research code for studying **Super Weights (SWs)** and **Under-Outliers (UOs)** as interpretable scalar mechanisms in Qwen3 language models.

The working hypothesis is that a small number of individual weight coordinates can have outsized, mechanistically meaningful effects on model behavior. Quantization is used here as a controlled stress test: when low-bit compression perturbs a model, the weights that dominate group ranges, preserve attention sinks, or recover perplexity become useful probes into the model's internal computation.

This repository should stay private while the SW/UO detection framework, calibration recipe, and interpretation claims are still being validated.

## Research Framing

This project is not mainly a deployment quantization repo. It uses quantization because it creates a measurable perturbation that exposes unusually important scalar weights. The near-term goal is a stronger detection framework for useful and not-useful weights under low-bit quantization.

The central questions are:

1. **Which individual weights are load-bearing?**
   Super Weights are rare scalar outliers, especially in `mlp.down_proj`, whose ablation can sharply damage model quality.

2. **Which weights distort local numerical geometry?**
   Under-Outliers sit at the max/min frontier of a quantization group. They can dominate the group range while contributing little to FP16 behavior, making them useful probes for separating numerical artifacts from functional computation.

3. **Can we classify weights by causal damage and quantization cost?**
   A Super Weight is high damage if removed and should be protected. A useful UO is low damage if removed but high cost if kept in the quantization grid.

4. **Do SW/UO interventions restore internal trajectories?**
   Depthwise diagnostics compare BOS token norm, attention sink score, and residual-stream compression across FP16, low-bit baseline, and UO-zeroed variants.

5. **Can scalar edits reveal mechanism rather than only improve PPL?**
   PPL, KL, and EAR are treated as external checks. The more interesting signal is whether small scalar interventions recover recognizable internal structure.

## What This Code Does

**Range-frontier UO detection** finds weights that determine a group's dynamic range. A candidate is admitted only if zeroing it is FP16-harmless and improves the quantized model.

**Super Weight detection** scans `mlp.down_proj` matrices for extreme scalar outliers that should be protected or studied rather than zeroed.

**Depthwise diagnostics** compare per-layer signals:

- BOS token hidden-state norm
- attention mass on the BOS token
- top singular-value fraction of the residual stream

These curves test whether an intervention changes the internal computation in a coherent way.

**Alpha sweeps** scale known Super Weights by `alpha` to test whether exact FP16 restoration is optimal, or whether the perturbed downstream computation prefers a different scalar value.

## Current Detector Direction

The intended detector is forward-only and endpoint-based:

1. Use the actual production quantization groups.
2. Shortlist range-frontier weights that affect group scale.
3. Reject candidates that damage FP16 behavior.
4. Admit only candidates that improve the final quantized model.
5. Veto or stress-test candidates near known SW routes, sink-token behavior, or long-context-sensitive features.

This is deliberately different from ranking weights only by magnitude, gradient, or Hessian proxy. The working criterion is:

```text
useful UO = FP16-harmless + quantization-useful
```

The framework can also be viewed as a residual decomposition:

```text
W_fp16 - W_q = SW-protected residual + UO-zeroed residual + normal quantized residual
```

The open question is whether the remaining residual has structure, such as channel concentration, low-rank directions, or sink-token-specific distortion.

## Why Quantization Appears Here

Low-bit group quantization gives a concrete way to expose scalar mechanisms:

```text
group scale = (max - min) / (2^bits - 1)
```

A single frontier weight can set this scale for many neighboring weights. If removing that scalar improves the quantized model without hurting FP16, it suggests the weight is numerically dominant but not functionally load-bearing in the original model.

That makes UOs complementary to SWs:

- **SWs**: functionally load-bearing scalar weights.
- **UOs**: numerically load-bearing scalar weights that may be safely removed under quantization.

The contrast is the interpretability target.

## Setup

```bash
git clone https://github.com/JiamingPan/llm-quant-experiments.git
cd llm-quant-experiments
pip install -r requirements.txt
```

On Great Lakes, the configs expect local Qwen3 model folders under scratch:

```text
/scratch/huterer_root/huterer0/jiamingp/models/qwen3-1b7
/scratch/huterer_root/huterer0/jiamingp/models/qwen3-8b
```

## Usage

**Detect UO candidates:**

```bash
python scripts/run_uo_detection.py --config configs/qwen3_1b7.yaml
python scripts/run_uo_detection.py --config configs/qwen3_8b.yaml
```

**Run depthwise interpretability diagnostics:**

```bash
python scripts/run_diagnostics.py \
    --config configs/qwen3_1b7.yaml \
    --uo-path results/uo_candidates_1b7.json \
    --output results/figures/depthwise_1b7.png
```

**Great Lakes SLURM:**

```bash
sbatch scripts/slurm/uo_1b7.sbatch
sbatch scripts/slurm/uo_8b.sbatch
```

## Repository Layout

```text
quant_analysis/
    quantize.py      low-bit group quantization used as the perturbation operator
    detect.py        SW and UO scalar candidate detection
    metrics.py       PPL, delta-PPL, KL, EAR, and admission gates
    diagnostics.py   BOS norm, attention sink score, residual compression curves
scripts/
    run_uo_detection.py   candidate discovery and admission pipeline
    run_diagnostics.py    depthwise comparison plots
    slurm/                Great Lakes batch scripts
configs/
    qwen3_1b7.yaml
    qwen3_8b.yaml
results/
    local outputs, ignored by git
```

## Admission Criterion

A UO candidate is admitted only if both conditions hold:

1. `delta_PPL_FP16 / baseline_PPL < 0.005`
2. Quantized-model PPL improves after zeroing the candidate

This gate is intentionally conservative: a candidate should be harmless in the original model and useful under the quantization perturbation.

## Privacy Note

This repository contains early experiment code, unpublished research framing, and detector strategy. Keep it private until the SW/UO detection results, calibration recipe, baselines, and interpretation claims are validated.

## References

- Yu et al. 2024 — Super Weights ([arXiv:2411.07191](https://arxiv.org/abs/2411.07191))
- Lin et al. 2024 — AWQ ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978))
- Guo et al. 2025 — PQI/ReQuant ([arXiv:2503.01901](https://arxiv.org/abs/2503.01901))
- Helcig et al. 2026 — SLQ ([arXiv:2605.02404](https://arxiv.org/abs/2605.02404))
- Queipo-de-Llano et al. 2026 — Mix-Compress-Refine ([arXiv:2510.06477](https://arxiv.org/abs/2510.06477))
