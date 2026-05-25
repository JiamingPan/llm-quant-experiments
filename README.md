# Weight Handles

This repository studies individual scalar weights as mechanistic handles on transformer computation. The working question is:

> Which single coordinates in a language model act as causal write-points into reusable internal features, and which large coordinates are visually extreme but functionally weak?

The code implements the experimental program behind **Weight Outliers as Information Handles**. It uses public Hugging Face models, forward passes, activation hooks, temporary scalar edits, and residual-stream patching to separate functional outliers from matched controls.

## Core Ideas

**Super Weights** are rare scalar coordinates whose deletion causes measurable behavioral damage. They are treated as positive controls for live internal routes.

**Under-Outliers** are large scalar coordinates whose deletion has low causal damage. They are treated as negative controls for separating magnitude from mechanism.

**Shadow Outliers** are coordinates that look weak alone but matter jointly with nearby or related coordinates.

**Silent Stabilizers** and **Obsolete Anchors** are low-damage coordinates that still leave structured residual traces, suggesting a possible stabilizing or historical role.

## Causal Feature-Handle Test

The main experiment has four stages:

1. **Candidate selection**
   - known Super Weight coordinates from config lists, following Yu et al. 2024 ([arXiv:2411.07191](https://arxiv.org/abs/2411.07191))
   - top robust-magnitude coordinates using `|w - median(W)| / (MAD(W) + eps)`
   - functional Under-Outlier candidates with low forward-pass damage
   - matched controls selected by layer, matrix type, row norm, column norm, and optional activation statistics

2. **Behavioral ablation**
   - zero one coordinate or a coordinate set
   - compare teacher and edited model distributions with conditional KL
   - report mean KL, tail/CVaR KL, top-token flip rate, logit-margin change, and perplexity change

3. **Residual-stream tracking**
   - hook transformer layers
   - measure the induced residual perturbation
   - compute the top principal direction and stability score `lambda_1 / (sum lambda + eps)`

4. **Patch-back and spreadability**
   - delete a coordinate
   - patch the lost residual perturbation back
   - test whether behavior is recovered by full residual patching, rank-one patching, nearby distributed edits, or channel scaling

## Prompt Strata

Experiments evaluate candidates across deterministic public prompt strata:

- prose
- code
- math
- chat
- long-context
- separators/BOS
- sentinel prompts

These strata are intentionally small by default so the pipeline is easy to inspect and extend.

## Setup

```bash
cd <repo>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the full Causal Feature-Handle Test:

```bash
python scripts/run_feature_handle_test.py \
    --config configs/qwen3_1b7.yaml \
    --output results/feature_handles_1b7.json
```

Track residual perturbations for the highest-damage candidates:

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

## Layout

```text
weight_handles/
    candidates.py     candidate selection
    ablation.py       coordinate and set ablation
    residual.py       residual-stream hooks and perturbation tracking
    patching.py       patch-back, rank-one patch, spreadability tests
    strata.py         deterministic public prompt strata
    metrics.py        KL, tail/CVaR KL, token flips, margins, perplexity
    taxonomy.py       intervention-result taxonomy
scripts/
    run_feature_handle_test.py
    run_residual_tracking.py
    make_report.py
configs/
    qwen3_1b7.yaml
    qwen3_8b.yaml
tests/
    unit tests for metrics, ablation, and patching
results/
    generated artifacts
```

## Reproducibility

The code is forward-pass only. It does not require gradient-based detectors. Candidate lists are deterministic under fixed seeds, YAML configs, and public model identifiers.

## References

- Yu et al. 2024, **Super Weights**, [arXiv:2411.07191](https://arxiv.org/abs/2411.07191)
