"""
Asymmetric per-group weight quantization.
No external quantization libraries — pure PyTorch.

TODO (Codex): implement the three functions below.
See the full spec in the Codex prompt.
"""

import torch
from typing import Optional


def quantize_tensor(
    W: torch.Tensor,
    bits: int = 2,
    group_size: int = 128,
) -> tuple:
    """
    Asymmetric per-group quantization of weight matrix W.

    Args:
        W:          (out_features, in_features) float tensor
        bits:       quantization bit-width (2 or 3 typical)
        group_size: number of columns per quantization group

    Returns:
        W_q:        dequantized float tensor, same shape as W
        scales:     (out_features, n_groups) float tensor
        zero_points:(out_features, n_groups) int tensor, clamped [0, 2^bits-1]
        W_int:      (out_features, in_features) int tensor [0, 2^bits-1]

    Formula per group g covering cols [g*gs : (g+1)*gs]:
        scale      = (max - min) / (2^bits - 1)
        zero_point = round(-min / scale), clamped [0, 2^bits-1]
        W_int      = clamp(round(W / scale) + zero_point, 0, 2^bits-1)
        W_dq       = (W_int - zero_point) * scale
    """
    raise NotImplementedError("TODO: implement in Codex")


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
        bits:        quantization bit-width
        group_size:  group size for per-group quantization
        layer_names: if given, only quantize these named layers

    Returns:
        dict mapping layer_name -> (W_q, scales, zero_points, W_int)
    """
    raise NotImplementedError("TODO: implement in Codex")


def compute_quantization_error(
    W_orig: torch.Tensor,
    W_q: torch.Tensor,
) -> dict:
    """
    Quantization error statistics between original and dequantized weights.

    Returns dict with keys:
        'mse':          mean squared error
        'max_abs_error':maximum absolute error
        'relative_mse': MSE / mean(W_orig^2)
    """
    raise NotImplementedError("TODO: implement in Codex")
