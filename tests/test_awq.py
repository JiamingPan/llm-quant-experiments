import torch

from awq import (
    awq_linear_weight,
    collect_linear_inputs,
    pseudo_quantize_tensor,
    search_awq_scale,
    temporary_awq,
)
from tests.helpers import ToyLM, ToyTokenizer


def test_pseudo_quantize_tensor_preserves_shape():
    W = torch.randn(7, 13)
    W_q = pseudo_quantize_tensor(W, n_bits=4, group_size=5)
    assert W_q.shape == W.shape
    assert W_q.dtype == W.dtype


def test_search_awq_scale_returns_weight_and_alpha():
    W = torch.randn(6, 8)
    X = torch.randn(16, 8)
    result = search_awq_scale(W, X, n_bits=4, group_size=4, n_grid=4)
    assert result["weight"].shape == W.shape
    assert 0.0 <= result["alpha"] <= 1.0
    assert result["loss"] >= 0.0


def test_awq_linear_weight_runs_on_linear():
    layer = torch.nn.Linear(8, 6, bias=False)
    X = torch.randn(16, 8)
    result = awq_linear_weight(layer, X, n_bits=4, group_size=4, n_grid=4)
    assert result["weight"].shape == layer.weight.shape


def test_collect_inputs_and_temporary_awq_restores_weight():
    model = ToyLM()
    tokenizer = ToyTokenizer()
    original = model.head.weight.data.detach().clone()
    buffers = collect_linear_inputs(
        model,
        tokenizer,
        ["abcdef"],
        layer_names=["head"],
        batch_size=1,
        max_length=8,
        max_tokens_per_layer=8,
        device="cpu",
    )
    assert "head" in buffers
    assert buffers["head"].shape[-1] == model.head.in_features

    with temporary_awq(model, buffers, layer_names=["head"], n_bits=4, group_size=4, n_grid=2):
        assert not torch.equal(model.head.weight.data, original)

    assert torch.equal(model.head.weight.data, original)
