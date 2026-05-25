import pytest
import torch

from tests.helpers import ToyLM, ToyTokenizer
from weight_handles.ablation import evaluate_coordinate_ablation, patch_weights, zero_coordinates


def test_patch_weights_restores_after_exception():
    model = ToyLM()
    original = model.head.weight.data[1, 2].detach().clone()

    with pytest.raises(RuntimeError):
        with patch_weights(model, {("head", 1, 2): 0.0}):
            assert model.head.weight.data[1, 2].item() == 0.0
            raise RuntimeError("stop")

    assert torch.equal(model.head.weight.data[1, 2], original)


def test_zero_coordinate_ablation_restores_weight():
    model = ToyLM()
    tokenizer = ToyTokenizer()
    original = model.head.weight.data[1, 2].detach().clone()

    result = evaluate_coordinate_ablation(
        model,
        tokenizer,
        ["abc", "def"],
        "head",
        1,
        2,
        batch_size=2,
        max_length=8,
        device="cpu",
    )

    assert result["n_tokens"] > 0
    assert "mean_kl" in result
    assert torch.equal(model.head.weight.data[1, 2], original)


def test_zero_coordinates_context_restores_weight():
    model = ToyLM()
    original = model.head.weight.data[1, 2].detach().clone()
    with zero_coordinates(model, [("head", 1, 2)]):
        assert model.head.weight.data[1, 2].item() == 0.0
    assert torch.equal(model.head.weight.data[1, 2], original)
