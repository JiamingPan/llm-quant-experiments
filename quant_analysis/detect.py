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
import torch.nn as nn
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
    if W.dim() != 2:
        raise ValueError("W must be a 2D tensor")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    W_float = W.detach().float()
    out_features, in_features = W_float.shape
    n_groups = (in_features + group_size - 1) // group_size
    candidates = []

    for group_idx in range(n_groups):
        start = group_idx * group_size
        end = min(start + group_size, in_features)
        group = W_float[:, start:end]
        width = end - start
        if width <= 1:
            continue

        max_vals, max_idx = group.max(dim=1)
        min_vals, min_idx = group.min(dim=1)
        original_range = max_vals - min_vals

        max_masked = group.clone()
        max_masked.scatter_(1, max_idx.view(-1, 1), float("-inf"))
        second_largest = max_masked.max(dim=1).values
        new_max = torch.maximum(second_largest, torch.zeros_like(second_largest))
        max_reduction = original_range - (new_max - min_vals)

        min_masked = group.clone()
        min_masked.scatter_(1, min_idx.view(-1, 1), float("inf"))
        second_smallest = min_masked.min(dim=1).values
        new_min = torch.minimum(second_smallest, torch.zeros_like(second_smallest))
        min_reduction = original_range - (max_vals - new_min)

        max_rows = torch.nonzero(max_reduction > 0, as_tuple=False).flatten()
        for row_tensor in max_rows:
            row = int(row_tensor.item())
            col = start + int(max_idx[row].item())
            candidates.append({
                "row": row,
                "col": col,
                "group": group_idx,
                "value": float(W_float[row, col].item()),
                "type": "max",
                "scale_reduction": float(max_reduction[row].item()),
            })

        min_rows = torch.nonzero(min_reduction > 0, as_tuple=False).flatten()
        for row_tensor in min_rows:
            row = int(row_tensor.item())
            col = start + int(min_idx[row].item())
            candidates.append({
                "row": row,
                "col": col,
                "group": group_idx,
                "value": float(W_float[row, col].item()),
                "type": "min",
                "scale_reduction": float(min_reduction[row].item()),
            })

    return sorted(candidates, key=lambda x: x["scale_reduction"], reverse=True)


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
    if packet_size <= 0:
        raise ValueError("packet_size must be positive")
    return [candidates[i:i + packet_size] for i in range(0, len(candidates), packet_size)]


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
    results = []

    for name, module in model.named_modules():
        if not name.endswith("mlp.down_proj") or not isinstance(module, nn.Linear):
            continue

        W = module.weight.data
        abs_W = W.abs().float()
        threshold = torch.quantile(abs_W.reshape(-1), percentile / 100.0)
        rows, cols = torch.nonzero(abs_W > threshold, as_tuple=True)

        for row_tensor, col_tensor in zip(rows, cols):
            row = int(row_tensor.item())
            col = int(col_tensor.item())
            value = float(W[row, col].item())
            percentile_rank = float((abs_W <= abs_W[row, col]).float().mean().item() * 100.0)
            results.append({
                "layer": name,
                "row": row,
                "col": col,
                "value": value,
                "percentile_rank": percentile_rank,
            })

    return sorted(results, key=lambda x: abs(x["value"]), reverse=True)


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
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    group_idx = col // group_size
    start = group_idx * group_size
    end = min(start + group_size, W.shape[1])
    group = W[row, start:end].detach().float()

    original_range = group.max() - group.min()
    patched_group = group.clone()
    patched_group[col - start] = 0.0
    new_range = patched_group.max() - patched_group.min()

    return float((original_range - new_range).item())
