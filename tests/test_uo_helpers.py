import torch

from weight_handles.uo import (
    evaluate_candidate,
    layer_slug,
    local_relative_error,
    temporary_weight_replacements,
    top_abs_candidate,
)


def test_local_relative_error_zero_when_weights_match():
    W = torch.randn(4, 6)
    X = torch.randn(8, 6)
    err, denom = local_relative_error(W, W, X)
    assert err == 0.0
    assert denom > 0.0


def test_top_abs_candidate_finds_largest_abs():
    W = torch.tensor([[0.1, -3.0], [2.0, 0.5]])
    assert top_abs_candidate(W) == (0, 1)


def test_evaluate_candidate_returns_expected_keys():
    W = torch.randn(4, 8)
    X = torch.randn(10, 8)
    record = evaluate_candidate(W, X, "layer.test", 0, 1, 2, "rule", group_size=4)
    assert record["layer"] == "layer.test"
    assert record["group"] == 0
    assert record["row"] == 1
    assert record["col"] == 2
    assert "B_g" in record
    assert "A_local_g" in record


def test_layer_slug_qwen_layer_name():
    assert layer_slug("model.layers.14.mlp.down_proj") == "layer14_down_proj"


def test_temporary_weight_replacements_restores_original():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
    original = model.get_submodule("0").weight.detach().clone()
    replacement = torch.ones_like(original)
    with temporary_weight_replacements(model, {"0": replacement}):
        assert torch.equal(model.get_submodule("0").weight, replacement)
    assert torch.equal(model.get_submodule("0").weight, original)
