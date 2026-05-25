# llm-quant-experiments

Experimental tools for studying **Super Weights (SWs)** and **Under-Outliers (UOs)** as scalar mechanisms in Qwen3 language models.

The core idea is interpretability-first: individual weight coordinates can sometimes act like unusually important control points in a transformer. This repository uses controlled perturbations, including low-bit group quantization and targeted scalar edits, to identify those weights and measure their effects on model behavior and internal activations.

Quantization is not the main claim of the project. It is a useful experimental stressor: when a perturbation changes the model, the weights that dominate group ranges, preserve attention-sink behavior, or recover perplexity become candidates for mechanistic study.

## Research Questions

1. **Which individual weights are load-bearing?**
   Super Weights are rare scalar outliers, often in `mlp.down_proj`, whose ablation can sharply damage model quality.

2. **Which large weights are not functionally important?**
   Under-Outliers can dominate the numerical range of a local group while contributing little to the FP16 model's behavior. They are useful negative controls for separating magnitude from mechanism.

3. **Can scalar edits explain internal trajectories?**
   The project compares depthwise signals such as BOS token norm, attention-sink score, and residual-stream compression before and after targeted scalar interventions.

4. **Can we distinguish useful from not-useful weights?**
   SWs and UOs can be viewed as opposite cases: SWs are high-causal-damage weights that should be protected, while useful UOs are low-causal-damage weights whose presence can make perturbations worse.

## What This Code Does

**Super Weight detection** scans model weights for rare scalar outliers, especially in `mlp.down_proj`, and tests whether they are causally important.

**Under-Outlier detection** finds range-frontier weights that affect the local quantization group scale, then admits a candidate only if deleting it is FP16-harmless and useful under the perturbation.

**Forward metrics** measure PPL, delta-PPL, KL divergence, and argmax agreement between original and perturbed models.

**Depthwise diagnostics** collect per-layer signals:

- BOS token hidden-state norm
- attention mass on the BOS token
- top singular-value fraction of the residual stream

**Alpha sweeps** scale selected Super Weights to test whether exact restoration is optimal, or whether the surrounding perturbed computation prefers a different scalar value.

## Conceptual Framing

The project treats a weight intervention as a probe:

```text
zero, preserve, or rescale one scalar -> measure behavioral and internal change
```

For UOs, the working admission rule is:

```text
useful UO = FP16-harmless + perturbation-useful
```

For SWs, the working rule is the opposite:

```text
super weight = high-damage scalar that must be protected or studied
```

This makes SWs and UOs a pair of positive and negative controls for studying information transport in LLMs. SWs point toward live scalar routes; UOs point toward large but weak or abandoned scalar routes that can still affect numerical representations.

## Why Quantization Appears Here

Low-bit group quantization gives a concrete way to expose scalar effects. For a group of weights:

```text
group scale = (max - min) / (2^bits - 1)
```

A single frontier scalar can set the scale for many neighboring weights. If removing that scalar leaves the FP16 model intact but improves the perturbed model, the scalar is probably numerically dominant without being functionally load-bearing.

This is why the repository contains quantization utilities. They are perturbation tools for interpretability experiments, not a full quantization library.

## Setup

```bash
git clone https://github.com/JiamingPan/llm-quant-experiments.git
cd llm-quant-experiments
pip install -r requirements.txt
```

The scripts use Hugging Face `AutoModelForCausalLM` and `AutoTokenizer`. `model.name` in each config can be either a Hugging Face model ID or a local model directory.

## Usage

**Detect UO candidates:**

```bash
python scripts/run_uo_detection.py --config configs/qwen3_1b7.yaml
python scripts/run_uo_detection.py --config configs/qwen3_8b.yaml
```

**Run depthwise diagnostics:**

```bash
python scripts/run_diagnostics.py \
    --config configs/qwen3_1b7.yaml \
    --uo-path results/uo_candidates_1b7.json \
    --output results/figures/depthwise_1b7.png
```

**SLURM:**

```bash
sbatch scripts/slurm/uo_1b7.sbatch
sbatch scripts/slurm/uo_8b.sbatch
```

Edit the config and SLURM paths for your local cluster environment before submitting jobs.

## Repository Layout

```text
quant_analysis/
    quantize.py      group quantization perturbations
    detect.py        SW and UO scalar candidate detection
    metrics.py       PPL, delta-PPL, KL, EAR, and admission gates
    diagnostics.py   BOS norm, attention sink score, residual compression curves
scripts/
    run_uo_detection.py   candidate discovery and admission pipeline
    run_diagnostics.py    depthwise comparison plots
    slurm/                example batch scripts
configs/
    qwen3_1b7.yaml
    qwen3_8b.yaml
results/
    local outputs, ignored by git
```

## Status

This is an exploratory research repo. The current code is meant to support experiments and generate evidence, not to present final claims about Super Weights, Under-Outliers, or transformer mechanisms.

## References

- Yu et al. 2024 — Super Weights ([arXiv:2411.07191](https://arxiv.org/abs/2411.07191))
- Lin et al. 2024 — AWQ ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978))
- Guo et al. 2025 — PQI/ReQuant ([arXiv:2503.01901](https://arxiv.org/abs/2503.01901))
- Helcig et al. 2026 — SLQ ([arXiv:2605.02404](https://arxiv.org/abs/2605.02404))
- Queipo-de-Llano et al. 2026 — Mix-Compress-Refine ([arXiv:2510.06477](https://arxiv.org/abs/2510.06477))
