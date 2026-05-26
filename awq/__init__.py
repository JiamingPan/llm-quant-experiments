"""
Small PyTorch AWQ reproduction utilities.

This module implements the calibration-time AWQ objective from Lin et al.,
arXiv:2306.00978, without external quantization libraries or inference kernels.
Weights are stored as dequantized tensors for analysis and notebook tests.
"""

from .core import (
    apply_awq_to_model,
    awq_linear_weight,
    collect_linear_inputs,
    pseudo_quantize_tensor,
    search_awq_scale,
    temporary_awq,
)

__all__ = [
    "apply_awq_to_model",
    "awq_linear_weight",
    "collect_linear_inputs",
    "pseudo_quantize_tensor",
    "search_awq_scale",
    "temporary_awq",
]
