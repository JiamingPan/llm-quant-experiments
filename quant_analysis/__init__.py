"""
quant_analysis: forward-pass LLM quantization analysis toolkit.

Modules:
    quantize    - asymmetric group quantization (W2/W3, group_size=128)
    detect      - range-frontier UO detection, super weight detection
    metrics     - delta-PPL, KL divergence, EAR, UO admission gating
    diagnostics - depthwise BOS norm, attention sink score, rep entropy
"""

from .quantize import quantize_tensor, quantize_model_weights, compute_quantization_error
from .detect import find_range_frontier_candidates, build_candidate_packets, detect_super_weights
from .metrics import compute_perplexity, compute_delta_ppl, compute_kl_divergence, compute_ear, gate_uo_candidate
from .diagnostics import compute_depthwise_diagnostics, plot_depthwise_comparison, compute_alpha_sweep

__version__ = "0.1.0"
__all__ = [
    "quantize_tensor", "quantize_model_weights", "compute_quantization_error",
    "find_range_frontier_candidates", "build_candidate_packets", "detect_super_weights",
    "compute_perplexity", "compute_delta_ppl", "compute_kl_divergence",
    "compute_ear", "gate_uo_candidate",
    "compute_depthwise_diagnostics", "plot_depthwise_comparison", "compute_alpha_sweep",
]
