"""
Deterministic candidate selection for scalar feature-handle experiments.

Known Super Weight coordinates can be supplied from config lists following
Yu et al. 2024, arXiv:2411.07191.
"""

import torch
import torch.nn as nn

from .ablation import coordinate_id, evaluate_coordinate_ablation


def matrix_type(layer_name):
    return layer_name.split(".")[-1]


def iter_linear_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def candidate_key(candidate):
    return f"{candidate['layer']}:{int(candidate['row'])}:{int(candidate['col'])}"


def known_super_weight_candidates(model, known_locations):
    """
    Load known Super Weight coordinates from a config list.
    """
    candidates = []
    for item in known_locations or []:
        layer_name = item["layer"]
        row = int(item["row"])
        col = int(item["col"])
        weight = model.get_submodule(layer_name).weight.detach()
        candidates.append({
            "layer": layer_name,
            "matrix_type": matrix_type(layer_name),
            "row": row,
            "col": col,
            "value": float(weight[row, col].float().item()),
            "score": float(item.get("score", 0.0)),
            "source": "known_super_weight",
            "reference": "Yu et al. 2024, arXiv:2411.07191",
        })
    return candidates


def robust_magnitude_scores(W, eps=1e-8):
    W_float = W.detach().float()
    median = torch.median(W_float)
    mad = torch.median(torch.abs(W_float - median))
    return torch.abs(W_float - median) / (mad + eps)


def top_magnitude_candidates(model, k=256, eps=1e-8, layer_names=None):
    """
    Select top scalar outliers by robust matrix-local magnitude score.
    """
    candidates = []
    allowed = set(layer_names) if layer_names is not None else None
    for layer_name, module in iter_linear_layers(model):
        if allowed is not None and layer_name not in allowed:
            continue

        W = module.weight.detach()
        scores = robust_magnitude_scores(W, eps=eps)
        n = min(k, scores.numel())
        if n == 0:
            continue
        values, indices = torch.topk(scores.reshape(-1), n)
        n_cols = W.shape[1]
        for score, index in zip(values, indices):
            flat = int(index.item())
            row = flat // n_cols
            col = flat % n_cols
            candidates.append({
                "layer": layer_name,
                "matrix_type": matrix_type(layer_name),
                "row": row,
                "col": col,
                "value": float(W[row, col].float().item()),
                "score": float(score.item()),
                "source": "robust_magnitude",
            })

    candidates.sort(key=lambda item: (-item["score"], item["layer"], item["row"], item["col"]))
    return candidates[:k]


def functional_under_outlier_candidates(
    model,
    tokenizer,
    texts,
    candidate_pool,
    max_candidates=128,
    max_mean_kl=1e-4,
    max_relative_ppl_change=0.001,
    batch_size=4,
    max_length=512,
    device="cuda",
):
    """
    Select large coordinates whose deletion has low measured causal damage.
    """
    accepted = []
    for candidate in candidate_pool:
        metrics = evaluate_coordinate_ablation(
            model,
            tokenizer,
            texts,
            candidate["layer"],
            int(candidate["row"]),
            int(candidate["col"]),
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        item = dict(candidate)
        item["source"] = "under_outlier_functional"
        item["ablation"] = metrics
        if (
            metrics["mean_kl"] <= max_mean_kl
            and abs(metrics["relative_ppl_change"]) <= max_relative_ppl_change
        ):
            accepted.append(item)
        if len(accepted) >= max_candidates:
            break
    return accepted


def _activation_distance(layer_name, row, col, target, activation_stats):
    if not activation_stats or layer_name not in activation_stats:
        return 0.0

    stats = activation_stats[layer_name]
    distance = 0.0
    if "row" in stats and target["row"] < len(stats["row"]) and row < len(stats["row"]):
        row_values = torch.as_tensor(stats["row"]).float()
        scale = row_values.std().item() + 1e-8
        distance += abs(float(row_values[row] - row_values[target["row"]])) / scale
    if "col" in stats and target["col"] < len(stats["col"]) and col < len(stats["col"]):
        col_values = torch.as_tensor(stats["col"]).float()
        scale = col_values.std().item() + 1e-8
        distance += abs(float(col_values[col] - col_values[target["col"]])) / scale
    return distance


def matched_control_candidates(model, targets, activation_stats=None, max_controls=None):
    """
    Match controls on layer, matrix type, row norm, column norm, and optional
    activation statistics.
    """
    controls = []
    used = {candidate_key(item) for item in targets}
    layer_modules = dict(iter_linear_layers(model))

    for target in targets:
        layer_name = target["layer"]
        if layer_name not in layer_modules:
            continue

        W = layer_modules[layer_name].weight.detach().float()
        row_norms = W.norm(dim=1)
        col_norms = W.norm(dim=0)
        row_scale = row_norms.std().item() + 1e-8
        col_scale = col_norms.std().item() + 1e-8
        target_row = int(target["row"])
        target_col = int(target["col"])

        row_distance = torch.abs(row_norms - row_norms[target_row]) / row_scale
        col_distance = torch.abs(col_norms - col_norms[target_col]) / col_scale
        distance = row_distance[:, None] + col_distance[None, :]

        flat_order = torch.argsort(distance.reshape(-1), stable=True)
        n_cols = W.shape[1]
        chosen = None
        for flat_index in flat_order:
            flat = int(flat_index.item())
            row = flat // n_cols
            col = flat % n_cols
            key = f"{layer_name}:{row}:{col}"
            if key in used:
                continue
            chosen = (row, col, key)
            break
        if chosen is None:
            continue

        row, col, key = chosen
        used.add(key)
        controls.append({
            "layer": layer_name,
            "matrix_type": matrix_type(layer_name),
            "row": row,
            "col": col,
            "value": float(W[row, col].item()),
            "score": float(distance[row, col].item() + _activation_distance(layer_name, row, col, target, activation_stats)),
            "source": "matched_control",
            "matched_to": coordinate_id(target),
        })
        if max_controls is not None and len(controls) >= max_controls:
            break

    return controls


def dedupe_candidates(candidates):
    seen = set()
    deduped = []
    for candidate in candidates:
        key = candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped
