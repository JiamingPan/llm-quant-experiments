"""
Asymmetric per-group low-bit weight mapping.
No external low-bit libraries — pure PyTorch.

TODO (Codex): implement the three functions below.
See the full spec in the Codex prompt.
"""

import torch
import torch.nn as nn
from typing import Optional


def quantize_tensor(
    W: torch.Tensor,
    bits: int = 2,
    group_size: int = 128,
) -> tuple:
    """
    Asymmetric per-group low-bit mapping of weight matrix W.

    Args:
        W:          (out_features, in_features) float tensor
        bits:       bit-width (2 or 3 typical)
        group_size: number of columns per local group

    Returns:
        W_q:        reconstructed float tensor, same shape as W
        scales:     (out_features, n_groups) float tensor
        zero_points:(out_features, n_groups) int tensor, clamped [0, 2^bits-1]
        W_int:      (out_features, in_features) int tensor [0, 2^bits-1]

    Formula per group g covering cols [g*gs : (g+1)*gs]:
        scale      = (max - min) / (2^bits - 1)
        zero_point = round(-min / scale), clamped [0, 2^bits-1]
        W_int      = clamp(round(W / scale) + zero_point, 0, 2^bits-1)
        W_dq       = (W_int - zero_point) * scale
    """
    if W.dim() != 2:
        raise ValueError("W must be a 2D tensor")
    if bits <= 0:
        raise ValueError("bits must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    W_float = W.detach().float()
    out_features, in_features = W_float.shape
    n_groups = (in_features + group_size - 1) // group_size
    qmax = (2 ** bits) - 1

    W_q = torch.empty_like(W_float)
    W_int = torch.empty(W_float.shape, device=W_float.device, dtype=torch.int64)
    scales = torch.empty((out_features, n_groups), device=W_float.device, dtype=W_float.dtype)
    zero_points = torch.empty((out_features, n_groups), device=W_float.device, dtype=torch.int64)

    for group_idx in range(n_groups):
        start = group_idx * group_size
        end = min(start + group_size, in_features)
        group = W_float[:, start:end]

        min_vals = group.min(dim=1, keepdim=True).values
        max_vals = group.max(dim=1, keepdim=True).values
        scale = (max_vals - min_vals) / qmax
        safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        zero_point = torch.clamp(torch.round(-min_vals / safe_scale), 0, qmax)
        group_int = torch.clamp(torch.round(group / safe_scale) + zero_point, 0, qmax)
        group_q = (group_int - zero_point) * safe_scale

        zero_scale = scale == 0
        if zero_scale.any():
            group_q = torch.where(zero_scale.expand_as(group_q), group, group_q)

        W_q[:, start:end] = group_q
        W_int[:, start:end] = group_int.to(torch.int64)
        scales[:, group_idx] = scale.squeeze(1)
        zero_points[:, group_idx] = zero_point.squeeze(1).to(torch.int64)

    return W_q.to(dtype=W.dtype), scales, zero_points, W_int


def quantize_model_weights(
    model,
    bits: int = 2,
    group_size: int = 128,
    layer_names: Optional[list] = None,
) -> dict:
    """
    Apply quantize_tensor to all Linear layers in model.
    Does NOT modify model in place.

    Args:
        model:       HuggingFace transformer model (eval mode)
        bits:        bit-width
        group_size:  group size for per-group mapping
        layer_names: if given, only quantize these named layers

    Returns:
        dict mapping layer_name -> (W_q, scales, zero_points, W_int)
    """
    selected = set(layer_names) if layer_names is not None else None
    results = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if selected is not None and name not in selected:
            continue
        results[name] = quantize_tensor(module.weight.data, bits=bits, group_size=group_size)

    return results


def compute_quantization_error(
    W_orig: torch.Tensor,
    W_q: torch.Tensor,
) -> dict:
    """
    Error statistics between original and reconstructed weights.

    Returns dict with keys:
        'mse':          mean squared error
        'max_abs_error':maximum absolute error
        'relative_mse': MSE / mean(W_orig^2)
    """
    diff = W_orig.float() - W_q.float()
    mse = diff.pow(2).mean()
    denom = W_orig.float().pow(2).mean()
    if denom.item() == 0:
        relative_mse = torch.tensor(0.0 if mse.item() == 0 else float("inf"), device=mse.device)
    else:
        relative_mse = mse / denom

    return {
        "mse": float(mse.item()),
        "max_abs_error": float(diff.abs().max().item()),
        "relative_mse": float(relative_mse.item()),
    }
