"""
Depthwise diagnostic curves for comparing FP16 vs quantized models.

Three signals per layer:
    bos_norm:    L2 norm of BOS token hidden state — tracks attention sink strength
    sink_score:  mean max attention weight on BOS across heads — direct sink measure
    rep_entropy: top singular value fraction of residual stream — compression proxy

These curves distinguish whether UO zeroing recovers PPL by actually restoring
the sink/compression trajectory, or just by accident.

Alpha sweep:
    Tests W_sw -> alpha * W_sw for alpha in {0.8, 0.9, 1.0, 1.1, 1.2}.
    Finds whether static FP16 restoration (alpha=1.0) is optimal or whether
    the quantized downstream Jacobian prefers a different scale.

TODO (Codex): implement the three functions below.
See the full spec in the Codex prompt.

IMPLEMENTATION NOTES:
    - Register hooks BEFORE forward pass, remove them AFTER.
    - Use output_attentions=True to get attention weights.
    - SVD via torch.linalg.svd on hidden_state[0, :, :].
    - BOS is always token index 0 (transformers prepend BOS by default).
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional


def _model_device(model, requested=None):
    try:
        param_device = next(model.parameters()).device
        if param_device.type != "meta":
            return param_device
    except StopIteration:
        pass

    if requested is not None:
        device = torch.device(requested)
        if device.type != "cuda" or torch.cuda.is_available():
            return device

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_depthwise_diagnostics(
    model,
    tokenizer,
    text: str,
    device: str = "cuda",
) -> dict:
    """
    Single forward pass collecting per-layer diagnostic signals.

    Args:
        model:     HuggingFace transformer (eval mode)
        tokenizer: corresponding tokenizer (add_special_tokens=True)
        text:      input string (BOS prepended automatically)
        device:    'cuda' or 'cpu'

    Returns:
        {
            'bos_norm':   list[float],  # length = n_layers
            'sink_score': list[float],  # length = n_layers
            'rep_entropy':list[float],  # length = n_layers
            'n_layers':   int,
            'n_tokens':   int,
        }

    Hook registration:
        Register on each model.layers[i] (or equivalent).
        Capture: hidden_state output (for bos_norm, rep_entropy).
        Use output_attentions=True for sink_score.
        Remove all hooks in finally block.
    """
    input_device = _model_device(model, device)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(input_device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(input_device)

    layers = model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers") else model.layers
    n_layers = len(layers)
    hidden_states = [None] * n_layers
    handles = []
    was_training = model.training
    model.eval()

    def make_hook(idx):
        def hook(module, inputs, output):
            hidden_state = output[0] if isinstance(output, (tuple, list)) else output
            hidden_states[idx] = hidden_state.detach()
        return hook

    try:
        for idx, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(idx)))

        kwargs = {"input_ids": input_ids, "output_attentions": True}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        with torch.no_grad():
            outputs = model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    attentions = outputs.attentions if hasattr(outputs, "attentions") else outputs[-1]
    bos_norm = []
    sink_score = []
    rep_entropy = []

    for idx in range(n_layers):
        hs = hidden_states[idx]
        attn = attentions[idx]

        bos_norm.append(float(hs[0, 0, :].float().norm().item()))

        bos_attention = attn[0, :, :, 0].float()
        sink_score.append(float(bos_attention.max(dim=-1).values.mean().item()))

        singular_values = torch.linalg.svd(hs[0].float(), full_matrices=False).S
        denom = singular_values.sum()
        if denom.item() == 0:
            rep_entropy.append(0.0)
        else:
            rep_entropy.append(float((singular_values[0] / denom).item()))

    return {
        "bos_norm": bos_norm,
        "sink_score": sink_score,
        "rep_entropy": rep_entropy,
        "n_layers": n_layers,
        "n_tokens": int(input_ids.shape[1]),
    }


def plot_depthwise_comparison(
    diagnostics_dict: dict,
    labels: list,
    save_path: Optional[str] = None,
) -> None:
    """
    3-panel figure comparing depthwise curves across model variants.

    Args:
        diagnostics_dict: {'fp16': diag, 'w2_baseline': diag, 'w2_uo': diag}
                          values are outputs of compute_depthwise_diagnostics
        labels:           ['FP16', 'W2 baseline', 'W2 + UO zeroed']
        save_path:        if given, save figure here; else plt.show()

    Layout:
        Row of 3 panels, shared x-axis (layer index):
        Panel 1: bos_norm vs layer
        Panel 2: sink_score vs layer
        Panel 3: rep_entropy vs layer
        Legend in panel 1.
    """
    signals = [
        ("bos_norm", "BOS Token Norm"),
        ("sink_score", "Attention Sink Score"),
        ("rep_entropy", "Representation Entropy (σ₁/Σσ)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    for ax, (signal, title) in zip(axes, signals):
        for (key, diagnostics), label in zip(diagnostics_dict.items(), labels):
            values = diagnostics[signal]
            ax.plot(range(len(values)), values, label=label)
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel(signal)

    axes[0].legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def compute_alpha_sweep(
    model,
    tokenizer,
    texts: list,
    sw_locations: list,
    alphas: list = None,
    baseline_ppl: Optional[float] = None,
) -> dict:
    """
    Test alpha scaling for super weight restoration.

    Temporarily scales each SW by alpha and measures PPL.
    Finds whether alpha=1.0 (FP16 restoration) is optimal.

    Args:
        model:        FP16 model (eval mode)
        tokenizer:    tokenizer
        texts:        calibration texts for PPL evaluation
        sw_locations: list of (layer_name, row, col, original_fp16_value)
        alphas:       list of scale factors (default [0.8, 0.9, 1.0, 1.1, 1.2])
        baseline_ppl: FP16 baseline PPL (computed if None)

    Returns:
        {alpha: {'ppl': float, 'delta_ppl': float} for alpha in alphas}

    Use try/finally to restore original SW values after each alpha test.
    """
    if alphas is None:
        alphas = [0.8, 0.9, 1.0, 1.1, 1.2]
    from .metrics import compute_delta_ppl, compute_perplexity

    if baseline_ppl is None:
        baseline_ppl = compute_perplexity(model, tokenizer, texts)

    results = {}
    for alpha in alphas:
        interventions = {}
        for layer_name, row, col, original_fp16_value in sw_locations:
            interventions[(layer_name, row, col)] = alpha * original_fp16_value

        delta = compute_delta_ppl(
            model,
            tokenizer,
            texts,
            interventions,
            baseline_ppl=baseline_ppl,
        )
        results[alpha] = {
            "ppl": float(delta["patched_ppl"]),
            "delta_ppl": float(delta["delta_ppl"]),
        }

    return results
