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
import math


def _model_device(model, requested=None):
    try:
        param_device = next(model.parameters()).device
        if param_device.type != "meta":
            return param_device
    except StopIteration:
        pass

    if requested is not None:
        device = torch.device(requested)
        if device.type != "cuda" or torch.cuda.is_available():
            return device

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _logits(outputs):
    return outputs.logits if hasattr(outputs, "logits") else outputs[0]


def _tokenized_batches(tokenizer, texts, batch_size, max_length, device):
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        if len(batch_texts) > 1 and tokenizer.pad_token_id is None:
            for text in batch_texts:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                yield input_ids, attention_mask
            continue

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=len(batch_texts) > 1,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        yield input_ids, attention_mask


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
    input_device = _model_device(model, device)
    was_training = model.training
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    try:
        with torch.no_grad():
            for input_ids, attention_mask in _tokenized_batches(
                tokenizer, texts, batch_size, max_length, input_device
            ):
                if input_ids.shape[1] < 2:
                    continue

                kwargs = {"input_ids": input_ids}
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask
                outputs = model(**kwargs)
                logits = _logits(outputs)

                shift_logits = logits[:, :-1, :].contiguous().float()
                labels = input_ids[:, 1:].contiguous()
                losses = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    labels.view(-1),
                    reduction="none",
                )

                if attention_mask is not None:
                    mask = attention_mask[:, 1:].contiguous().view(-1).float()
                    total_nll += float((losses * mask).sum().item())
                    total_tokens += int(mask.sum().item())
                else:
                    total_nll += float(losses.sum().item())
                    total_tokens += int(labels.numel())
    finally:
        model.train(was_training)

    if total_tokens == 0:
        return float("inf")
    return float(math.exp(total_nll / total_tokens))


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
    if baseline_ppl is None:
        baseline_ppl = compute_perplexity(model, tokenizer, texts)

    originals = []
    patched_ppl = None

    try:
        with torch.no_grad():
            for (layer_name, row, col), new_value in weight_interventions.items():
                weight = model.get_submodule(layer_name).weight.data
                original_value = weight[row, col].detach().clone()
                originals.append((weight, row, col, original_value))
                weight[row, col] = torch.as_tensor(new_value, device=weight.device, dtype=weight.dtype)

        patched_ppl = compute_perplexity(model, tokenizer, texts)
    finally:
        with torch.no_grad():
            for weight, row, col, original_value in reversed(originals):
                weight[row, col].copy_(original_value)

    delta_ppl = patched_ppl - baseline_ppl
    relative_delta = delta_ppl / baseline_ppl if baseline_ppl != 0 else float("inf")
    return {
        "baseline_ppl": float(baseline_ppl),
        "patched_ppl": float(patched_ppl),
        "delta_ppl": float(delta_ppl),
        "relative_delta": float(relative_delta),
    }


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
    device_fp16 = _model_device(model_fp16)
    device_q = _model_device(model_q)
    was_training_fp16 = model_fp16.training
    was_training_q = model_q.training
    model_fp16.eval()
    model_q.eval()

    total_kl = 0.0
    total_tokens = 0

    try:
        with torch.no_grad():
            for input_ids, attention_mask in _tokenized_batches(
                tokenizer, texts, batch_size, max_length, device_fp16
            ):
                input_ids_q = input_ids.to(device_q)
                attention_mask_q = attention_mask.to(device_q) if attention_mask is not None else None

                kwargs_fp16 = {"input_ids": input_ids}
                kwargs_q = {"input_ids": input_ids_q}
                if attention_mask is not None:
                    kwargs_fp16["attention_mask"] = attention_mask
                    kwargs_q["attention_mask"] = attention_mask_q

                logits_fp16 = _logits(model_fp16(**kwargs_fp16)).float()
                logits_q = _logits(model_q(**kwargs_q)).to(logits_fp16.device).float()

                p = F.softmax(logits_fp16, dim=-1)
                log_q = F.log_softmax(logits_q, dim=-1)
                token_kl = F.kl_div(log_q, p, reduction="none").sum(dim=-1)

                if attention_mask is not None:
                    mask = attention_mask.to(token_kl.device).float()
                    total_kl += float((token_kl * mask).sum().item())
                    total_tokens += int(mask.sum().item())
                else:
                    total_kl += float(token_kl.sum().item())
                    total_tokens += int(token_kl.numel())
    finally:
        model_fp16.train(was_training_fp16)
        model_q.train(was_training_q)

    if total_tokens == 0:
        return float("inf")
    return float(total_kl / total_tokens)


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
    device_fp16 = _model_device(model_fp16)
    device_q = _model_device(model_q)
    was_training_fp16 = model_fp16.training
    was_training_q = model_q.training
    model_fp16.eval()
    model_q.eval()

    agreed = 0.0
    total_tokens = 0

    try:
        with torch.no_grad():
            for input_ids, attention_mask in _tokenized_batches(
                tokenizer, texts, batch_size, max_length, device_fp16
            ):
                input_ids_q = input_ids.to(device_q)
                attention_mask_q = attention_mask.to(device_q) if attention_mask is not None else None

                kwargs_fp16 = {"input_ids": input_ids}
                kwargs_q = {"input_ids": input_ids_q}
                if attention_mask is not None:
                    kwargs_fp16["attention_mask"] = attention_mask
                    kwargs_q["attention_mask"] = attention_mask_q

                pred_fp16 = _logits(model_fp16(**kwargs_fp16)).argmax(dim=-1)
                pred_q = _logits(model_q(**kwargs_q)).argmax(dim=-1).to(pred_fp16.device)
                matches = (pred_fp16 == pred_q).float()

                if attention_mask is not None:
                    mask = attention_mask.to(matches.device).float()
                    agreed += float((matches * mask).sum().item())
                    total_tokens += int(mask.sum().item())
                else:
                    agreed += float(matches.sum().item())
                    total_tokens += int(matches.numel())
    finally:
        model_fp16.train(was_training_fp16)
        model_q.train(was_training_q)

    if total_tokens == 0:
        return 0.0
    return float(agreed / total_tokens)


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
    delta_result = compute_delta_ppl(
        model,
        tokenizer,
        texts,
        {(layer_name, row, col): 0.0},
        baseline_ppl=baseline_ppl,
    )
    delta_ppl_fp16 = delta_result["delta_ppl"]
    relative_fp16_damage = delta_ppl_fp16 / baseline_ppl if baseline_ppl != 0 else float("inf")

    model_q = getattr(model, "_quant_analysis_model_q", model)
    weight = model_q.get_submodule(layer_name).weight.data
    original_value = weight[row, col].detach().clone()

    try:
        with torch.no_grad():
            weight[row, col] = torch.as_tensor(0.0, device=weight.device, dtype=weight.dtype)
        new_q_ppl = compute_perplexity(model_q, tokenizer, texts)
    finally:
        with torch.no_grad():
            weight[row, col].copy_(original_value)

    q_improvement = model_q_ppl - new_q_ppl
    admitted = relative_fp16_damage < fp16_budget and q_improvement > 0
    if admitted:
        reason = "admitted"
    elif relative_fp16_damage >= fp16_budget:
        reason = "rejected: FP16 damage exceeds budget"
    else:
        reason = "rejected: quantized PPL did not improve"

    return {
        "admitted": bool(admitted),
        "delta_ppl_fp16": float(delta_ppl_fp16),
        "relative_fp16_damage": float(relative_fp16_damage),
        "q_improvement": float(q_improvement),
        "reason": reason,
    }
