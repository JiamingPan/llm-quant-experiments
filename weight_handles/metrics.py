"""
Forward-pass metrics for scalar weight intervention experiments.
"""

import math

import torch
import torch.nn.functional as F


def model_device(model, requested=None):
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


def logits_from_output(output):
    return output.logits if hasattr(output, "logits") else output[0]


def tokenized_batches(tokenizer, texts, batch_size=4, max_length=512, device=None):
    if isinstance(texts, str):
        texts = [texts]

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        if len(batch_texts) > 1 and getattr(tokenizer, "pad_token_id", None) is None:
            for text in batch_texts:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                yield input_ids, attention_mask
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


def lm_loss_from_logits(logits, input_ids, attention_mask=None):
    if input_ids.shape[1] < 2:
        return 0.0, 0

    shift_logits = logits[:, :-1, :].contiguous().float()
    labels = input_ids[:, 1:].contiguous()
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        labels.view(-1),
        reduction="none",
    )

    if attention_mask is not None:
        mask = attention_mask[:, 1:].contiguous().view(-1).float()
        return float((losses * mask).sum().item()), int(mask.sum().item())

    return float(losses.sum().item()), int(labels.numel())


def compute_perplexity(model, tokenizer, texts, batch_size=4, max_length=512, device="cuda"):
    """
    Compute standard next-token perplexity over a text list.
    """
    input_device = model_device(model, device)
    was_training = model.training
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    try:
        with torch.no_grad():
            for input_ids, attention_mask in tokenized_batches(
                tokenizer, texts, batch_size=batch_size, max_length=max_length, device=input_device
            ):
                kwargs = {"input_ids": input_ids}
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask
                logits = logits_from_output(model(**kwargs))
                nll, n_tokens = lm_loss_from_logits(logits, input_ids, attention_mask)
                total_nll += nll
                total_tokens += n_tokens
    finally:
        model.train(was_training)

    if total_tokens == 0:
        return float("inf")
    return float(math.exp(total_nll / total_tokens))


def _valid_position_mask(input_ids, attention_mask=None):
    if input_ids.shape[1] < 2:
        return torch.zeros_like(input_ids[:, :0], dtype=torch.bool)
    if attention_mask is None:
        return torch.ones(input_ids.shape[0], input_ids.shape[1] - 1, device=input_ids.device, dtype=torch.bool)
    return attention_mask[:, :-1].bool()


def token_kl_values(teacher_logits, student_logits, input_ids, attention_mask=None):
    """
    Return KL(p_teacher || p_student) for each valid token position.
    """
    mask = _valid_position_mask(input_ids, attention_mask)
    if mask.numel() == 0 or int(mask.sum().item()) == 0:
        return teacher_logits.new_zeros((0,), dtype=torch.float32)

    teacher = teacher_logits[:, :-1, :].float()
    student = student_logits[:, :-1, :].float()
    p_teacher = F.softmax(teacher, dim=-1)
    log_student = F.log_softmax(student, dim=-1)
    kl = F.kl_div(log_student, p_teacher, reduction="none").sum(dim=-1)
    return kl[mask].detach().float()


def top_token_flip_values(teacher_logits, student_logits, input_ids, attention_mask=None):
    mask = _valid_position_mask(input_ids, attention_mask)
    if mask.numel() == 0 or int(mask.sum().item()) == 0:
        return teacher_logits.new_zeros((0,), dtype=torch.float32)

    teacher_top = teacher_logits[:, :-1, :].argmax(dim=-1)
    student_top = student_logits[:, :-1, :].argmax(dim=-1)
    return (teacher_top[mask] != student_top[mask]).detach().float()


def logit_margin_values(logits, input_ids, attention_mask=None):
    mask = _valid_position_mask(input_ids, attention_mask)
    if mask.numel() == 0 or int(mask.sum().item()) == 0:
        return logits.new_zeros((0,), dtype=torch.float32)

    top2 = torch.topk(logits[:, :-1, :].float(), k=2, dim=-1).values
    margins = top2[..., 0] - top2[..., 1]
    return margins[mask].detach().float()


def empty_comparison_stats():
    return {
        "kl_values": [],
        "flip_sum": 0.0,
        "margin_change_sum": 0.0,
        "teacher_nll": 0.0,
        "student_nll": 0.0,
        "n_tokens": 0,
    }


def update_comparison_stats(stats, teacher_logits, student_logits, input_ids, attention_mask=None):
    kl = token_kl_values(teacher_logits, student_logits, input_ids, attention_mask)
    flips = top_token_flip_values(teacher_logits, student_logits, input_ids, attention_mask)
    teacher_margin = logit_margin_values(teacher_logits, input_ids, attention_mask)
    student_margin = logit_margin_values(student_logits, input_ids, attention_mask)
    teacher_nll, n_tokens = lm_loss_from_logits(teacher_logits, input_ids, attention_mask)
    student_nll, _ = lm_loss_from_logits(student_logits, input_ids, attention_mask)

    stats["kl_values"].append(kl.cpu())
    stats["flip_sum"] += float(flips.sum().item())
    if teacher_margin.numel() > 0:
        stats["margin_change_sum"] += float((student_margin - teacher_margin).sum().item())
    stats["teacher_nll"] += teacher_nll
    stats["student_nll"] += student_nll
    stats["n_tokens"] += n_tokens
    return stats


def tail_cvar(values, tail_fraction=0.05):
    values = values.float().flatten()
    if values.numel() == 0:
        return 0.0
    k = max(1, int(math.ceil(values.numel() * tail_fraction)))
    return float(torch.topk(values, k).values.mean().item())


def finalize_comparison_stats(stats, tail_fraction=0.05):
    n_tokens = int(stats["n_tokens"])
    if stats["kl_values"]:
        kl_values = torch.cat(stats["kl_values"])
    else:
        kl_values = torch.empty(0)

    if n_tokens == 0:
        teacher_ppl = float("inf")
        student_ppl = float("inf")
        mean_kl = 0.0
        flip_rate = 0.0
        margin_change = 0.0
    else:
        teacher_ppl = float(math.exp(stats["teacher_nll"] / n_tokens))
        student_ppl = float(math.exp(stats["student_nll"] / n_tokens))
        mean_kl = float(kl_values.mean().item()) if kl_values.numel() else 0.0
        flip_rate = float(stats["flip_sum"] / n_tokens)
        margin_change = float(stats["margin_change_sum"] / n_tokens)

    delta_ppl = student_ppl - teacher_ppl
    relative_ppl_change = delta_ppl / teacher_ppl if teacher_ppl not in (0.0, float("inf")) else 0.0
    return {
        "mean_kl": mean_kl,
        "tail_cvar_kl": tail_cvar(kl_values, tail_fraction=tail_fraction),
        "top_token_flip_rate": flip_rate,
        "mean_logit_margin_change": margin_change,
        "teacher_ppl": teacher_ppl,
        "student_ppl": student_ppl,
        "delta_ppl": float(delta_ppl),
        "relative_ppl_change": float(relative_ppl_change),
        "n_tokens": n_tokens,
    }


def compare_logits(teacher_logits, student_logits, input_ids, attention_mask=None, tail_fraction=0.05):
    stats = empty_comparison_stats()
    update_comparison_stats(stats, teacher_logits, student_logits, input_ids, attention_mask)
    return finalize_comparison_stats(stats, tail_fraction=tail_fraction)


def compute_kl_divergence(model_teacher, model_student, tokenizer, texts, batch_size=4, max_length=512, device="cuda"):
    """
    Mean teacher-student KL over valid next-token positions.
    """
    metrics = compute_teacher_student_metrics(
        model_teacher,
        model_student,
        tokenizer,
        texts,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    return metrics["mean_kl"]


def compute_teacher_student_metrics(
    model_teacher,
    model_student,
    tokenizer,
    texts,
    batch_size=4,
    max_length=512,
    device="cuda",
    tail_fraction=0.05,
):
    """
    Compare two models on the same text stream using forward-pass distributions.
    """
    teacher_device = model_device(model_teacher, device)
    student_device = model_device(model_student, device)
    teacher_was_training = model_teacher.training
    student_was_training = model_student.training
    model_teacher.eval()
    model_student.eval()
    stats = empty_comparison_stats()

    try:
        with torch.no_grad():
            for input_ids, attention_mask in tokenized_batches(
                tokenizer, texts, batch_size=batch_size, max_length=max_length, device=teacher_device
            ):
                kwargs = {"input_ids": input_ids}
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask
                teacher_logits = logits_from_output(model_teacher(**kwargs))

                student_input_ids = input_ids.to(student_device)
                student_attention_mask = attention_mask.to(student_device) if attention_mask is not None else None
                student_kwargs = {"input_ids": student_input_ids}
                if student_attention_mask is not None:
                    student_kwargs["attention_mask"] = student_attention_mask
                student_logits = logits_from_output(model_student(**student_kwargs)).to(teacher_logits.device)
                update_comparison_stats(stats, teacher_logits, student_logits, input_ids, attention_mask)
    finally:
        model_teacher.train(teacher_was_training)
        model_student.train(student_was_training)

    return finalize_comparison_stats(stats, tail_fraction=tail_fraction)
