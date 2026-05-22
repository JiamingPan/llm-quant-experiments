"""
UO (under-outlier) and super weight detection.

UO detection strategy:
    Range-frontier: a weight is a UO candidate if it sits at the max or min
    of its quantization group (i.e., it determines the group scale).
    Zeroing it would reduce the group dynamic range and improve quantization
    for all other weights in the group.

Super weight detection:
    Extreme scalar outliers in mlp.down_proj layers specifically.
    These are load-bearing: ablating them collapses perplexity.

TODO (Codex): implement the four functions below.
See the full spec in the Codex prompt.
"""

import torch
from typing import Optional


def find_range_frontier_candidates(
    W: torch.Tensor,
    group_size: int = 128,
) -> list:
    """
    Find range-frontier UO candidates in weight matrix W.

    A candidate is any weight that is the current group max or min,
    such that zeroing it would reduce (max_val - min_val) > 0.

    Args:
        W:          (out_features, in_features) float tensor
        group_size: columns per quantization group

    Returns:
        list of dicts sorted by scale_reduction descending:
        {
            'row':             int,
            'col':             int,
            'group':           int,
            'value':           float,
            'type':            'max' | 'min',
            'scale_reduction': float,  # reduction in group scale if zeroed
        }
    """
    raise NotImplementedError("TODO: implement in Codex")


def build_candidate_packets(
    candidates: list,
    packet_size: int = 1,
) -> list:
    """
    Group candidates into packets for batch admission testing.

    Args:
        candidates:  list of candidate dicts from find_range_frontier_candidates
        packet_size: number of candidates per packet

    Returns:
        list of lists of candidate dicts
    """
    raise NotImplementedError("TODO: implement in Codex")


def detect_super_weights(
    model,
    percentile: float = 99.99,
) -> list:
    """
    Detect extreme outlier scalars in all mlp.down_proj layers.

    Args:
        model:      HuggingFace transformer model
        percentile: threshold percentile of abs(W) across the full matrix

    Returns:
        list of dicts sorted by abs(value) descending:
        {
            'layer':          str,   # e.g. 'model.layers.0.mlp.down_proj'
            'row':            int,
            'col':            int,
            'value':          float,
            'percentile_rank':float,
        }
    """
    raise NotImplementedError("TODO: implement in Codex")


def compute_group_scale_reduction(
    W: torch.Tensor,
    row: int,
    col: int,
    group_size: int = 128,
) -> float:
    """
    Compute reduction in group scale if W[row, col] is zeroed.

    Returns:
        original_scale - new_scale  (positive = UO is beneficial)
    """
    raise NotImplementedError("TODO: implement in Codex")
