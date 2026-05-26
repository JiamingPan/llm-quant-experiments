"""
Local RFIC scoring helpers shared by v0/v1 scripts.
"""

from typing import Any

import torch

from awq.core import pseudo_quantize_tensor


def local_relative_error(
    W_ref: torch.Tensor,
    W_test: torch.Tensor,
    X: torch.Tensor,
    eta: float = 1e-8,
) -> tuple[float, float]:
    assert W_ref.ndim == 2
    assert W_test.shape == W_ref.shape
    assert X.ndim == 2
    assert X.shape[1] == W_ref.shape[1]

    y_ref = X.float() @ W_ref.float().t()
    y_diff = X.float() @ (W_test.float() - W_ref.float()).t()
    numerator = y_diff.pow(2).sum(dim=1).mean()
    denominator = y_ref.pow(2).sum(dim=1).mean() + eta
    return float((numerator / denominator).item()), float(denominator.item())


def zero_coordinate(W_group: torch.Tensor, row: int, local_col: int) -> torch.Tensor:
    assert W_group.ndim == 2
    assert 0 <= row < W_group.shape[0]
    assert 0 <= local_col < W_group.shape[1]

    W_zero = W_group.clone()
    W_zero[row, local_col] = 0.0
    return W_zero


def evaluate_candidate(
    W: torch.Tensor,
    X: torch.Tensor,
    layer_name: str,
    group_index: int,
    row: int,
    local_col: int,
    rule: str,
    group_size: int = 128,
    n_bits: int = 4,
    zero_point: bool = True,
    eta: float = 1e-8,
) -> dict[str, Any]:
    start = group_index * group_size
    end = min(start + group_size, W.shape[1])
    W_group = W[:, start:end].detach().float()
    X_group = X[:, start:end].detach().float()
    assert W_group.shape[1] == X_group.shape[1]
    assert 0 <= local_col < W_group.shape[1]
    assert 0 <= row < W_group.shape[0]

    col = start + local_col
    weight_value = float(W_group[row, local_col].item())
    max_abs_before = float(W_group.abs().max().item())
    W_zero = zero_coordinate(W_group, row, local_col)
    max_abs_after = float(W_zero.abs().max().item())

    W_q_empty = pseudo_quantize_tensor(W_group, n_bits=n_bits, group_size=group_size, zero_point=zero_point).float()
    W_q_zero = pseudo_quantize_tensor(W_zero, n_bits=n_bits, group_size=group_size, zero_point=zero_point).float()
    E_empty, denom = local_relative_error(W_group, W_q_empty, X_group, eta=eta)
    E_zero, _ = local_relative_error(W_group, W_q_zero, X_group, eta=eta)
    A_local, _ = local_relative_error(W_group, W_zero, X_group, eta=eta)
    B_g = E_empty - E_zero

    return {
        "layer": layer_name,
        "rule": rule,
        "group": int(group_index),
        "row": int(row),
        "col": int(col),
        "local_col": int(local_col),
        "weight_value": weight_value,
        "weight_magnitude": float(abs(weight_value)),
        "group_start_col": int(start),
        "group_end_col": int(end),
        "max_abs_before": max_abs_before,
        "max_abs_after": max_abs_after,
        "E_empty": float(E_empty),
        "E_zeroed": float(E_zero),
        "B_g": float(B_g),
        "A_local_g": float(A_local),
        "score_g": float(B_g / (A_local + eta)),
        "denominator": float(denom),
        "n_bits": int(n_bits),
        "group_size": int(group_size),
        "zero_point": bool(zero_point),
    }


def top_abs_candidate(W_group: torch.Tensor) -> tuple[int, int]:
    assert W_group.ndim == 2
    flat_index = int(torch.argmax(W_group.abs()).item())
    return flat_index // W_group.shape[1], flat_index % W_group.shape[1]


def layer_slug(layer_name: str) -> str:
    parts = layer_name.split(".")
    if len(parts) >= 5 and parts[0] == "model" and parts[1] == "layers":
        return f"layer{parts[2]}_{parts[-1]}"
    return layer_name.replace(".", "_")
