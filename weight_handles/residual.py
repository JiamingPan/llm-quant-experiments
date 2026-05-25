"""
Residual-stream hooks and perturbation tracking for scalar interventions.
"""

import torch

from .ablation import zero_coordinates
from .metrics import model_device, tokenized_batches


def transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Could not locate transformer layers on model")


def _hidden_from_output(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def collect_residual_streams(
    model,
    tokenizer,
    texts,
    coordinates=None,
    batch_size=1,
    max_length=512,
    device="cuda",
):
    """
    Collect per-layer residual-stream states with hooks.
    """
    input_device = model_device(model, device)
    layers = transformer_layers(model)
    streams = [[] for _ in range(len(layers))]
    handles = []
    was_training = model.training
    model.eval()

    def make_hook(index):
        def hook(module, inputs, output):
            hidden = _hidden_from_output(output).detach().float().cpu()
            hidden = hidden.reshape(-1, hidden.shape[-1])
            streams[index].append(hidden)
        return hook

    try:
        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(index)))

        context = zero_coordinates(model, coordinates) if coordinates else None
        if context is None:
            with torch.no_grad():
                for input_ids, attention_mask in tokenized_batches(
                    tokenizer, texts, batch_size=batch_size, max_length=max_length, device=input_device
                ):
                    kwargs = {"input_ids": input_ids}
                    if attention_mask is not None:
                        kwargs["attention_mask"] = attention_mask
                    model(**kwargs)
        else:
            with context:
                with torch.no_grad():
                    for input_ids, attention_mask in tokenized_batches(
                        tokenizer, texts, batch_size=batch_size, max_length=max_length, device=input_device
                    ):
                        kwargs = {"input_ids": input_ids}
                        if attention_mask is not None:
                            kwargs["attention_mask"] = attention_mask
                        model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return [torch.cat(layer_streams, dim=0) if layer_streams else torch.empty(0) for layer_streams in streams]


def residual_perturbations(
    model,
    tokenizer,
    texts,
    coordinates,
    batch_size=1,
    max_length=512,
    device="cuda",
):
    """
    Return base-minus-ablated residual perturbations for each layer.
    """
    base = collect_residual_streams(
        model,
        tokenizer,
        texts,
        coordinates=None,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    ablated = collect_residual_streams(
        model,
        tokenizer,
        texts,
        coordinates=coordinates,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )

    deltas = []
    for base_layer, ablated_layer in zip(base, ablated):
        if base_layer.shape != ablated_layer.shape:
            n = min(base_layer.shape[0], ablated_layer.shape[0])
            deltas.append(base_layer[:n] - ablated_layer[:n])
        else:
            deltas.append(base_layer - ablated_layer)
    return deltas


def principal_component(delta, eps=1e-12):
    """
    Compute the top principal direction and spectral concentration score.
    """
    if delta.numel() == 0:
        return {
            "top_pc": torch.empty(0),
            "eigenvalues": torch.empty(0),
            "stability": 0.0,
        }

    X = delta.reshape(-1, delta.shape[-1]).float()
    X = X - X.mean(dim=0, keepdim=True)
    if X.shape[0] == 1:
        top_pc = X[0]
        norm = top_pc.norm() + eps
        return {
            "top_pc": top_pc / norm,
            "eigenvalues": torch.tensor([float((X * X).sum().item())]),
            "stability": 1.0,
        }

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    eigenvalues = S.pow(2)
    stability = float((eigenvalues[0] / (eigenvalues.sum() + eps)).item()) if eigenvalues.numel() else 0.0
    return {
        "top_pc": Vh[0].detach(),
        "eigenvalues": eigenvalues.detach(),
        "stability": stability,
    }


def stability_score(delta, eps=1e-12):
    return principal_component(delta, eps=eps)["stability"]


def track_residual_perturbation(
    model,
    tokenizer,
    texts,
    coordinates,
    batch_size=1,
    max_length=512,
    device="cuda",
):
    """
    Summarize residual perturbation strength and direction stability by layer.
    """
    deltas = residual_perturbations(
        model,
        tokenizer,
        texts,
        coordinates,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    summaries = []
    for layer_index, delta in enumerate(deltas):
        pc = principal_component(delta)
        summaries.append({
            "layer": layer_index,
            "delta_norm": float(delta.norm().item()) if delta.numel() else 0.0,
            "mean_token_delta_norm": float(delta.reshape(-1, delta.shape[-1]).norm(dim=-1).mean().item()) if delta.numel() else 0.0,
            "stability": float(pc["stability"]),
        })
    return {"layers": summaries, "n_layers": len(summaries)}
