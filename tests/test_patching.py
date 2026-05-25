import torch

from tests.helpers import ToyLM, ToyTokenizer
from weight_handles.patching import patch_back_sufficiency, rank_one_delta, residual_patch
from weight_handles.residual import residual_perturbations, transformer_layers


def test_rank_one_delta_preserves_shape():
    delta = torch.randn(2, 4, 6)
    patched = rank_one_delta(delta)
    assert patched.shape == delta.shape


def test_residual_patch_hooks_are_removed():
    model = ToyLM()
    layers = transformer_layers(model)
    before = len(layers[0]._forward_hooks)
    delta = [torch.zeros(1, 3, 8), torch.zeros(1, 3, 8)]
    with residual_patch(model, delta):
        assert len(layers[0]._forward_hooks) == before + 1
    assert len(layers[0]._forward_hooks) == before


def test_patch_back_sufficiency_runs_on_toy_model():
    model = ToyLM()
    tokenizer = ToyTokenizer()
    texts = ["abcdef"]
    coordinate = ("head", 1, 2)
    deltas = residual_perturbations(
        model,
        tokenizer,
        texts,
        [coordinate],
        batch_size=1,
        max_length=8,
        device="cpu",
    )
    result = patch_back_sufficiency(
        model,
        tokenizer,
        texts,
        [coordinate],
        layer_deltas=deltas,
        batch_size=1,
        max_length=8,
        device="cpu",
    )
    assert "recovery_ratio" in result
    assert torch.isfinite(torch.tensor(result["recovery_ratio"]))
