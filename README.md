# Weight-Outlier Measurement Harness

Forward-pass tools for testing whether scalar weight outliers explain downstream behavior in quantized language models.

**Suggested GitHub About:** Negative result and measurement harness for Super Weight / Under-Outlier analysis, AWQ calibration, and trustworthy quantization evaluation.

## Result

The Super Weight / Under-Outlier direction is closed as a clean negative result in this repo.

Across the experiments that motivated this codebase, the highest-sensitivity scalar weight outliers captured only a small fraction of the available reasoning improvement, roughly 1-6%. The useful signal was real but too small to explain the downstream accuracy gap. This means scalar outlier sensitivity, weight error, and layer reconstruction objectives are not reliable predictors of downstream task accuracy by themselves.

The implication is practical: quantization work needs task-aware calibration and trustworthy evaluation, not only local reconstruction loss or weight-error metrics. This repository is useful because it records that negative result and provides reusable measurement code for checking whether a proposed low-level proxy actually tracks model behavior.

## Toolkit

The code is forward-pass only. It uses public Hugging Face models, PyTorch hooks, temporary weight edits, and deterministic YAML configs.

Main components:

- **Behavioral ablation:** zero one coordinate or a coordinate set, then compare teacher and edited-model distributions with conditional KL, tail/CVaR KL, top-token flip rate, logit-margin change, and perplexity change.
- **Prompt strata:** evaluate candidate behavior on deterministic public prompt groups: prose, code, math, chat, long-context, separators/BOS, and sentinel prompts.
- **Residual-stream perturbation tracking:** hook transformer layers, measure the induced residual perturbation, and compute an SVD-based stability score `lambda_1 / (sum lambda + eps)`.
- **Patch-back tests:** delete coordinates, patch back the lost residual perturbation, and test full residual patching, rank-one patching, distributed edits, and channel scaling.
- **Candidate selection:** load known Super Weight coordinates, select robust magnitude outliers, find low-damage functional Under-Outlier candidates, and construct matched controls by layer, matrix type, row norm, column norm, and optional activation statistics.
- **AWQ reproduction:** a small from-scratch PyTorch implementation of activation-aware weight scaling and groupwise 4-bit round-to-nearest quantization for analysis runs. It stores dequantized tensors and does not use external quantization libraries or packed inference kernels.
- **RFIC/UO checks:** local range-frontier experiments and a full-model PPL check for testing whether locally safe candidate masks help a quantized model. In the current runs, the masks were safe but did not improve full-model AWQ PPL.

## Reproduce

Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the scalar outlier measurement harness:

```bash
python scripts/run_feature_handle_test.py \
    --config configs/qwen3_1b7.yaml \
    --output results/feature_handles_1b7.json
```

Track residual perturbations for selected candidates:

```bash
python scripts/run_residual_tracking.py \
    --config configs/qwen3_1b7.yaml \
    --candidates results/feature_handles_1b7.json \
    --output results/residual_tracking_1b7.json
```

Create a taxonomy table and summary figure:

```bash
python scripts/make_report.py \
    --input results/feature_handles_1b7.json \
    --table results/tables/feature_handle_taxonomy_1b7.csv \
    --figure results/figures/feature_handle_taxonomy_1b7.png
```

Run the RFIC local candidate screens and full-model PPL check:

```bash
python scripts/run_uo_v1.py --config configs/uo_v1.yaml
python scripts/run_uo_full_model.py --K 20 --ruleset B
```

Configs are provided for Qwen3-1.7B and Qwen3-8B:

```text
configs/qwen3_1b7.yaml
configs/qwen3_8b.yaml
```

The notebooks currently provide Qwen3-1.7B AWQ smoke-test and walkthrough runs:

```text
notebooks/awq_qwen3_1b7_smoke.ipynb
notebooks/awq_qwen3_1b7_detailed_walkthrough.ipynb
```

On Great Lakes, use the SLURM wrappers in `scripts/slurm/` and keep model paths in config files rather than hardcoding them in scripts.

## Repository Layout

```text
weight_handles/
    candidates.py     Super Weight, magnitude, functional UO, and matched-control candidate selection
    ablation.py       coordinate and set ablation with teacher-student metrics
    residual.py       residual-stream hooks and SVD perturbation tracking
    patching.py       patch-back, rank-one patch, and spreadability tests
    strata.py         deterministic public prompt strata
    metrics.py        KL, tail/CVaR KL, token flips, margins, and perplexity
    taxonomy.py       intervention-result taxonomy
    uo.py             local RFIC/UO scoring helpers
awq/
    core.py           plain PyTorch AWQ search and groupwise quantization utilities
scripts/
    run_feature_handle_test.py
    run_residual_tracking.py
    make_report.py
    run_uo_v0.py
    run_uo_v1.py
    run_uo_full_model.py
    make_uo_v1_zoom.py
configs/
    qwen3_1b7.yaml
    qwen3_8b.yaml
    uo_v1.yaml
tests/
    unit tests for metrics, ablation, patching, AWQ, and UO helpers
results/
    generated artifacts
```

## Technical Blog

This repository also includes a minimal Astro technical blog. Posts are Markdown files in `src/content/posts`, with frontmatter for title, date, summary, and tags. LaTeX math is rendered with KaTeX through Astro's Markdown pipeline.

Create a new draft:

```bash
npm run new-post "My Post Title"
```

Local development:

```bash
npm install
npm run dev
```

Build the static site:

```bash
npm run build
```

## Reproducibility

The measurement code is forward-pass only and does not require gradient-based detectors. Candidate lists are deterministic under fixed seeds, YAML configs, and public model identifiers. Generated artifacts should be written under `results/`.

## References

- Yu et al. 2024, **Super Weights**, [arXiv:2411.07191](https://arxiv.org/abs/2411.07191)
- Lin et al. 2024, **AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration**, [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
