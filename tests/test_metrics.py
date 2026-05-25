import torch

from tests.helpers import ToyLM, ToyTokenizer
from weight_handles.metrics import (
    compare_logits,
    compute_kl_divergence,
    compute_perplexity,
    compute_teacher_student_metrics,
)


def test_identical_models_have_zero_kl():
    model = ToyLM()
    tokenizer = ToyTokenizer()
    metrics = compute_teacher_student_metrics(
        model,
        model,
        tokenizer,
        ["abc"],
        batch_size=1,
        max_length=8,
        device="cpu",
    )

    assert metrics["mean_kl"] < 1e-7
    assert metrics["top_token_flip_rate"] == 0.0


def test_compute_kl_divergence_wrapper():
    model = ToyLM()
    tokenizer = ToyTokenizer()
    kl = compute_kl_divergence(model, model, tokenizer, ["abc"], batch_size=1, max_length=8, device="cpu")
    assert kl < 1e-7


def test_perplexity_is_finite():
    model = ToyLM()
    tokenizer = ToyTokenizer()
    ppl = compute_perplexity(model, tokenizer, ["abc"], batch_size=1, max_length=8, device="cpu")
    assert ppl > 0.0
    assert torch.isfinite(torch.tensor(ppl))


def test_compare_logits_reports_flip_rate():
    input_ids = torch.tensor([[1, 2, 3]])
    teacher = torch.zeros(1, 3, 5)
    student = torch.zeros(1, 3, 5)
    teacher[:, :, 1] = 2.0
    student[:, :, 2] = 2.0

    result = compare_logits(teacher, student, input_ids)
    assert result["top_token_flip_rate"] == 1.0
    assert result["mean_kl"] > 0.0
