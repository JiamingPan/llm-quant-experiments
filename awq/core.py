"""
Activation-aware weight scaling for low-bit weight-only analysis.

The implementation follows the AWQ search objective:

    min_s || Q(W diag(s)) diag(s)^-1 X - W X ||

with the paper's one-dimensional search space s = s_X^alpha, where s_X is the
mean absolute input activation per channel. This file intentionally stores
dequantized weights for experiments instead of packing integer kernels.
"""

from contextlib import contextmanager

import torch
import torch.nn as nn


def _model_device(model, requested=None):
    try:
        device = next(model.parameters()).device
        if device.type != "meta":
            return device
    except StopIteration:
        pass
    if requested is not None:
        device = torch.device(requested)
        if device.type != "cuda" or torch.cuda.is_available():
            return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _logits(output):
    return output.logits if hasattr(output, "logits") else output[0]


def _tokenized_batches(tokenizer, texts, batch_size, max_length, device):
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        if len(batch_texts) > 1 and getattr(tokenizer, "pad_token_id", None) is None:
            for text in batch_texts:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                attention_mask = enc.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                yield enc["input_ids"].to(device), attention_mask
            continue

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=len(batch_texts) > 1,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        yield input_ids, attention_mask


def pseudo_quantize_tensor(W, n_bits=4, group_size=128, zero_point=True):
    """
    Groupwise round-to-nearest quantization returning dequantized weights.

    Groups are contiguous chunks along the input-channel dimension.
    """
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    orig_shape = W.shape
    W_float = W.detach().float()
    if W_float.dim() != 2:
        W_flat = W_float.reshape(-1, W_float.shape[-1])
    else:
        W_flat = W_float

    out_features, in_features = W_flat.shape
    W_q = torch.empty_like(W_flat)

    if zero_point:
        qmin = 0
        qmax = (2 ** n_bits) - 1
    else:
        qmax = (2 ** (n_bits - 1)) - 1
        qmin = -qmax

    for start in range(0, in_features, group_size):
        end = min(start + group_size, in_features)
        group = W_flat[:, start:end]

        if zero_point:
            min_val = group.min(dim=1, keepdim=True).values
            max_val = group.max(dim=1, keepdim=True).values
            scale = (max_val - min_val) / max(qmax - qmin, 1)
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            zp = torch.clamp(torch.round(qmin - min_val / scale), qmin, qmax)
            q = torch.clamp(torch.round(group / scale) + zp, qmin, qmax)
            W_q[:, start:end] = (q - zp) * scale
        else:
            max_abs = group.abs().max(dim=1, keepdim=True).values
            scale = max_abs / max(qmax, 1)
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            q = torch.clamp(torch.round(group / scale), qmin, qmax)
            W_q[:, start:end] = q * scale

    return W_q.reshape(orig_shape).to(dtype=W.dtype, device=W.device)


def _activation_scales(X, eps=1e-8):
    scales = X.detach().float().abs().mean(dim=0).clamp(min=eps)
    return scales


def _normalize_scales(scales, eps=1e-8):
    scales = scales.clamp(min=eps)
    denom = torch.sqrt(scales.max() * scales.min()).clamp(min=eps)
    return scales / denom


def _linear_mse(X, W_ref, W_test, chunk_size=256):
    total = 0.0
    count = 0
    for start in range(0, X.shape[0], chunk_size):
        end = min(start + chunk_size, X.shape[0])
        x = X[start:end].float()
        y_ref = torch.matmul(x, W_ref.float().t())
        y_test = torch.matmul(x, W_test.float().t())
        total += float(torch.sum((y_ref - y_test).pow(2)).item())
        count += y_ref.numel()
    return total / max(count, 1)


def search_awq_scale(
    W,
    X,
    n_bits=4,
    group_size=128,
    zero_point=True,
    n_grid=20,
    eps=1e-8,
    chunk_size=256,
):
    """
    Search alpha in [0, 1] for s = mean(abs(X)) ** alpha.
    """
    if X.dim() > 2:
        X = X.reshape(-1, X.shape[-1])
    X = X.detach().float().to(W.device)
    W_float = W.detach().float()

    act = _activation_scales(X, eps=eps).to(W.device)
    best = None
    grid = torch.linspace(0.0, 1.0, steps=n_grid + 1)
    for alpha in grid:
        scales = _normalize_scales(act.pow(float(alpha.item())), eps=eps)
        W_scaled = W_float * scales.view(1, -1)
        W_q_scaled = pseudo_quantize_tensor(
            W_scaled,
            n_bits=n_bits,
            group_size=group_size,
            zero_point=zero_point,
        ).float()
        W_awq = W_q_scaled / scales.view(1, -1)
        loss = _linear_mse(X, W_float, W_awq, chunk_size=chunk_size)
        if best is None or loss < best["loss"]:
            best = {
                "alpha": float(alpha.item()),
                "loss": float(loss),
                "scales": scales.detach().cpu(),
                "weight": W_awq.to(dtype=W.dtype).detach().cpu(),
            }

    return best


def awq_linear_weight(
    module,
    inputs,
    n_bits=4,
    group_size=128,
    zero_point=True,
    n_grid=20,
):
    """
    Return AWQ dequantized weight and search metadata for one nn.Linear.
    """
    if not isinstance(module, nn.Linear):
        raise TypeError("module must be nn.Linear")
    result = search_awq_scale(
        module.weight.detach(),
        inputs,
        n_bits=n_bits,
        group_size=group_size,
        zero_point=zero_point,
        n_grid=n_grid,
    )
    result["weight"] = result["weight"].to(device=module.weight.device, dtype=module.weight.dtype)
    return result


def collect_linear_inputs(
    model,
    tokenizer,
    texts,
    layer_names=None,
    batch_size=1,
    max_length=512,
    max_tokens_per_layer=512,
    device="cuda",
):
    """
    Collect flattened input activations for selected Linear modules.
    """
    input_device = _model_device(model, device)
    selected = set(layer_names) if layer_names is not None else None
    buffers = {}
    handles = []
    was_training = model.training
    model.eval()

    def make_hook(name):
        def hook(module, inputs):
            if name in buffers and buffers[name].shape[0] >= max_tokens_per_layer:
                return
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).float().cpu()
            remaining = max_tokens_per_layer - buffers.get(name, torch.empty(0)).shape[0]
            if remaining <= 0:
                return
            x = x[:remaining]
            if name not in buffers:
                buffers[name] = x
            else:
                buffers[name] = torch.cat([buffers[name], x], dim=0)
        return hook

    try:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and (selected is None or name in selected):
                handles.append(module.register_forward_pre_hook(make_hook(name)))

        with torch.no_grad():
            for input_ids, attention_mask in _tokenized_batches(
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

    return buffers


def apply_awq_to_model(
    model,
    input_buffers,
    layer_names=None,
    n_bits=4,
    group_size=128,
    zero_point=True,
    n_grid=20,
):
    """
    Replace selected Linear weights with AWQ dequantized weights.
    """
    selected = set(layer_names) if layer_names is not None else set(input_buffers.keys())
    results = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if name not in selected or name not in input_buffers:
                continue
            if not isinstance(module, nn.Linear):
                continue
            result = awq_linear_weight(
                module,
                input_buffers[name],
                n_bits=n_bits,
                group_size=group_size,
                zero_point=zero_point,
                n_grid=n_grid,
            )
            module.weight.data.copy_(result["weight"])
            results[name] = {
                "alpha": result["alpha"],
                "loss": result["loss"],
                "n_inputs": int(input_buffers[name].shape[0]),
            }
    return results


@contextmanager
def temporary_awq(model, input_buffers, **kwargs):
    """
    Apply AWQ weights inside a context and restore originals in finally.
    """
    selected = set(kwargs.get("layer_names") or input_buffers.keys())
    originals = {}
    try:
        with torch.no_grad():
            for name, module in model.named_modules():
                if name in selected and isinstance(module, nn.Linear):
                    originals[name] = module.weight.data.detach().clone()
            results = apply_awq_to_model(model, input_buffers, **kwargs)
        yield results
    finally:
        with torch.no_grad():
            for name, module in model.named_modules():
                if name in originals:
                    module.weight.data.copy_(originals[name].to(module.weight.device))
