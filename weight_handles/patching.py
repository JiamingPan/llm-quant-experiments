"""
Patch-back and spreadability tests for scalar weight-outlier interventions.
"""

from contextlib import contextmanager

import torch

from .ablation import evaluate_set_ablation, normalize_coordinate, patch_weights, zero_coordinates
from .metrics import (
    empty_comparison_stats,
    finalize_comparison_stats,
    logits_from_output,
    model_device,
    tokenized_batches,
    update_comparison_stats,
)
from .residual import principal_component, residual_perturbations, transformer_layers


def rank_one_delta(delta):
    if delta.numel() == 0:
        return delta
    pc = principal_component(delta)["top_pc"].to(delta.device)
    if pc.numel() == 0:
        return torch.zeros_like(delta)
    flat = delta.reshape(-1, delta.shape[-1]).float()
    projection = (flat @ pc.float()).unsqueeze(-1) * pc.float().unsqueeze(0)
    return projection.reshape_as(delta).to(delta.dtype)


@contextmanager
def residual_patch(model, layer_deltas, rank_one=False):
    """
    Add stored residual perturbations to layer outputs, removing hooks in finally.
    """
    layers = transformer_layers(model)
    handles = []

    def make_hook(index):
        def hook(module, inputs, output):
            if index >= len(layer_deltas):
                return output
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            delta = layer_deltas[index]
            if delta.numel() == 0:
                return output
            if rank_one:
                delta = rank_one_delta(delta)
            delta = delta.to(device=hidden.device, dtype=hidden.dtype)
            if delta.dim() == 2:
                needed = hidden.shape[0] * hidden.shape[1]
                if delta.shape[0] >= needed:
                    delta = delta[:needed].reshape(hidden.shape[0], hidden.shape[1], hidden.shape[2])
                else:
                    delta = delta.unsqueeze(0)
            delta = delta[:hidden.shape[0], :hidden.shape[1], :hidden.shape[2]]
            patched = hidden.clone()
            patched[:, :delta.shape[1], :] = patched[:, :delta.shape[1], :] + delta
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            if isinstance(output, list):
                return [patched] + list(output[1:])
            return patched
        return hook

    try:
        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(index)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _patched_student_metrics(
    model,
    tokenizer,
    texts,
    coordinates,
    layer_deltas,
    rank_one=False,
    batch_size=1,
    max_length=512,
    device="cuda",
    tail_fraction=0.05,
):
    input_device = model_device(model, device)
    was_training = model.training
    model.eval()
    stats = empty_comparison_stats()

    try:
        offsets = [0 for _ in layer_deltas]
        for input_ids, attention_mask in tokenized_batches(
            tokenizer, texts, batch_size=batch_size, max_length=max_length, device=input_device
        ):
            kwargs = {"input_ids": input_ids}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask

            with torch.no_grad():
                teacher_logits = logits_from_output(model(**kwargs))

            token_count = input_ids.shape[0] * input_ids.shape[1]
            batch_deltas = []
            for index, delta in enumerate(layer_deltas):
                if delta.dim() == 2:
                    start = offsets[index]
                    end = start + token_count
                    batch_deltas.append(delta[start:end])
                    offsets[index] = end
                else:
                    batch_deltas.append(delta)

            with zero_coordinates(model, coordinates):
                with residual_patch(model, batch_deltas, rank_one=rank_one):
                    with torch.no_grad():
                        student_logits = logits_from_output(model(**kwargs))

            update_comparison_stats(stats, teacher_logits, student_logits, input_ids, attention_mask)
    finally:
        model.train(was_training)

    return finalize_comparison_stats(stats, tail_fraction=tail_fraction)


def patch_back_sufficiency(
    model,
    tokenizer,
    texts,
    coordinates,
    layer_deltas=None,
    batch_size=1,
    max_length=512,
    device="cuda",
):
    """
    Delete coordinates, patch the lost residual perturbation back, and report recovery.
    """
    damage = evaluate_set_ablation(
        model,
        tokenizer,
        texts,
        coordinates,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    if layer_deltas is None:
        layer_deltas = residual_perturbations(
            model,
            tokenizer,
            texts,
            coordinates,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )

    patched = _patched_student_metrics(
        model,
        tokenizer,
        texts,
        coordinates,
        layer_deltas,
        rank_one=False,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    recovery = (damage["mean_kl"] - patched["mean_kl"]) / (damage["mean_kl"] + 1e-12)
    return {
        "damage": damage,
        "patched": patched,
        "recovery_ratio": float(recovery),
    }


def rank_one_patch_sufficiency(
    model,
    tokenizer,
    texts,
    coordinates,
    layer_deltas=None,
    batch_size=1,
    max_length=512,
    device="cuda",
):
    if layer_deltas is None:
        layer_deltas = residual_perturbations(
            model,
            tokenizer,
            texts,
            coordinates,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
    damage = evaluate_set_ablation(
        model,
        tokenizer,
        texts,
        coordinates,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    patched = _patched_student_metrics(
        model,
        tokenizer,
        texts,
        coordinates,
        layer_deltas,
        rank_one=True,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    recovery = (damage["mean_kl"] - patched["mean_kl"]) / (damage["mean_kl"] + 1e-12)
    return {
        "damage": damage,
        "patched": patched,
        "recovery_ratio": float(recovery),
    }


@contextmanager
def nearby_weight_patch(model, coordinate, radius=4):
    layer_name, row, col = normalize_coordinate(coordinate)
    weight = model.get_submodule(layer_name).weight.detach()
    start = max(0, col - radius)
    end = min(weight.shape[1], col + radius + 1)
    neighbors = [idx for idx in range(start, end) if idx != col]
    if not neighbors:
        with zero_coordinates(model, [(layer_name, row, col)]):
            yield
        return

    value = float(weight[row, col].float().item())
    interventions = {(layer_name, row, col): 0.0}
    share = value / len(neighbors)
    for neighbor_col in neighbors:
        interventions[(layer_name, row, neighbor_col)] = float(weight[row, neighbor_col].float().item()) + share

    with patch_weights(model, interventions):
        yield


@contextmanager
def channel_scaling_patch(model, coordinate, scale=1.01):
    layer_name, row, col = normalize_coordinate(coordinate)
    weight = model.get_submodule(layer_name).weight.detach()
    interventions = {(layer_name, row, col): 0.0}
    for neighbor_col in range(weight.shape[1]):
        if neighbor_col == col:
            continue
        interventions[(layer_name, row, neighbor_col)] = float(weight[row, neighbor_col].float().item()) * scale
    with patch_weights(model, interventions):
        yield


def _metrics_with_context(model, tokenizer, texts, context_factory, batch_size=1, max_length=512, device="cuda"):
    input_device = model_device(model, device)
    was_training = model.training
    model.eval()
    stats = empty_comparison_stats()
    try:
        for input_ids, attention_mask in tokenized_batches(
            tokenizer, texts, batch_size=batch_size, max_length=max_length, device=input_device
        ):
            kwargs = {"input_ids": input_ids}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            with torch.no_grad():
                teacher_logits = logits_from_output(model(**kwargs))
            with context_factory():
                with torch.no_grad():
                    student_logits = logits_from_output(model(**kwargs))
            update_comparison_stats(stats, teacher_logits, student_logits, input_ids, attention_mask)
    finally:
        model.train(was_training)
    return finalize_comparison_stats(stats)


def spreadability_tests(
    model,
    tokenizer,
    texts,
    coordinate,
    batch_size=1,
    max_length=512,
    device="cuda",
    recovery_threshold=0.8,
):
    """
    Test whether a scalar's behavior can be recovered by distributed alternatives.
    """
    coordinates = [coordinate]
    damage = evaluate_set_ablation(
        model,
        tokenizer,
        texts,
        coordinates,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    layer_deltas = residual_perturbations(
        model,
        tokenizer,
        texts,
        coordinates,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )

    nearby = _metrics_with_context(
        model,
        tokenizer,
        texts,
        lambda: nearby_weight_patch(model, coordinate),
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    channel = _metrics_with_context(
        model,
        tokenizer,
        texts,
        lambda: channel_scaling_patch(model, coordinate),
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    residual_full = patch_back_sufficiency(
        model,
        tokenizer,
        texts,
        coordinates,
        layer_deltas=layer_deltas,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    residual_rank_one = rank_one_patch_sufficiency(
        model,
        tokenizer,
        texts,
        coordinates,
        layer_deltas=layer_deltas,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )

    def recovery(metrics):
        return float((damage["mean_kl"] - metrics["mean_kl"]) / (damage["mean_kl"] + 1e-12))

    attempts = {
        "nearby_weights": {"metrics": nearby, "recovery_ratio": recovery(nearby)},
        "channel_scaling": {"metrics": channel, "recovery_ratio": recovery(channel)},
        "residual_direction": residual_full,
        "rank_one_residual": residual_rank_one,
    }
    for item in attempts.values():
        item["recovered"] = bool(item["recovery_ratio"] >= recovery_threshold)

    return {
        "damage": damage,
        "attempts": attempts,
    }
