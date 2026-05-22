"""
Evaluation metrics for quantization analysis.

Key metrics:
    - Perplexity (PPL): standard language model quality metric
    - Delta-PPL: PPL change under weight interventions (UO zeroing / SW scaling)
    - KL divergence: KL(FP16 || quantized) averaged over tokens
    - EAR (Expected Acceptance Rate): fraction of tokens where FP16 and
      quantized model agree on argmax. EAR >= 0.99 ~ distribution-lossless.

UO admission gate:
    Admit candidate if:
        (1) relative FP16 PPL damage < fp16_budget  (FP16-harmless)
        (2) quantized model PPL improves after zeroing (Q-useful)

IMPORTANT implementation note for compute_delta_ppl:
    Save original weight value, patch, compute PPL, restore.
    Use try/finally to guarantee restoration even on error.

TODO (Codex): implement all functions below.
See the full spec in the Codex prompt.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_perplexity(
    model,
    tokenizer,
    texts: list,
    batch_size: int = 4,
    max_length: int = 512,
    device: str = "cuda",
) -> float:
    """
    Compute perplexity on a list of text strings.

    Use model.eval() and torch.no_grad().
    Truncate each text to max_length tokens.
    Return exp(mean NLL per token across all texts).
    """
    raise NotImplementedError("TODO: implement in Codex")


def compute_delta_ppl(
    model,
    tokenizer,
    texts: list,
    weight_interventions: dict,
    baseline_ppl: Optional[float] = None,
) -> dict:
    """
    Measure PPL change under temporary weight patches.

    Args:
        model:               HuggingFace model (FP16, eval mode)
        tokenizer:           corresponding tokenizer
        texts:               calibration texts
        weight_interventions:{(layer_name, row, col): new_value}
                              new_value=0.0 for UO zeroing
        baseline_ppl:        if None, compute it here

    Returns:
        {
            'baseline_ppl':  float,
            'patched_ppl':   float,
            'delta_ppl':     float,   # patched - baseline
            'relative_delta':float,   # delta_ppl / baseline_ppl
        }

    CRITICAL: use try/finally to restore original weights.
    """
    raise NotImplementedError("TODO: implement in Codex")


def compute_kl_divergence(
    model_fp16,
    model_q,
    tokenizer,
    texts: list,
    batch_size: int = 4,
    max_length: int = 512,
) -> float:
    """
    KL(p_fp16 || p_q) averaged over all tokens and texts.

    Use F.kl_div with log_target=False.
    Return mean KL per token (nats).
    """
    raise NotImplementedError("TODO: implement in Codex")


def compute_ear(
    model_fp16,
    model_q,
    tokenizer,
    texts: list,
    batch_size: int = 4,
    max_length: int = 512,
) -> float:
    """
    Expected Acceptance Rate: P(argmax_fp16 == argmax_q).

    Returns float in [0, 1].
    EAR >= 0.99 is the distribution-lossless threshold (SLQ paper).
    """
    raise NotImplementedError("TODO: implement in Codex")


def gate_uo_candidate(
    model,
    tokenizer,
    texts: list,
    layer_name: str,
    row: int,
    col: int,
    baseline_ppl: float,
    model_q_ppl: float,
    fp16_budget: float = 0.005,
) -> dict:
    """
    Admission test for a single UO candidate (forward-only).

    Admit if both conditions hold:
        (1) relative FP16 PPL damage < fp16_budget
        (2) quantized model PPL improves after zeroing

    Args:
        model:       FP16 model
        layer_name:  e.g. 'model.layers.3.mlp.down_proj'
        row, col:    weight coordinate
        baseline_ppl:FP16 PPL before any intervention
        model_q_ppl: quantized model PPL before zeroing this candidate
        fp16_budget: max allowable relative damage (default 0.5%)

    Returns:
        {
            'admitted':            bool,
            'delta_ppl_fp16':      float,
            'relative_fp16_damage':float,
            'q_improvement':       float,  # model_q_ppl - new_q_ppl (positive=better)
            'reason':              str,    # human-readable admission/rejection reason
        }
    """
    raise NotImplementedError("TODO: implement in Codex")
