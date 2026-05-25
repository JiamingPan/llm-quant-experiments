"""
Temporary scalar edits and forward-only behavioral ablation tests.
"""

from contextlib import contextmanager

import torch

from .metrics import (
    empty_comparison_stats,
    finalize_comparison_stats,
    logits_from_output,
    model_device,
    tokenized_batches,
    update_comparison_stats,
)


def normalize_coordinate(coordinate):
    if isinstance(coordinate, dict):
        return coordinate["layer"], int(coordinate["row"]), int(coordinate["col"])
    return coordinate[0], int(coordinate[1]), int(coordinate[2])


def coordinate_id(coordinate):
    layer, row, col = normalize_coordinate(coordinate)
    return f"{layer}:{row}:{col}"


@contextmanager
def patch_weights(model, interventions):
    """
    Temporarily set selected scalar weights, restoring all originals in finally.
    """
    originals = []
    try:
        with torch.no_grad():
            for key, new_value in interventions.items():
                layer_name, row, col = normalize_coordinate(key)
                weight = model.get_submodule(layer_name).weight
                original = weight.data[row, col].detach().clone()
                originals.append((weight, row, col, original))
                weight.data[row, col] = torch.as_tensor(new_value, device=weight.device, dtype=weight.dtype)
        yield
    finally:
        with torch.no_grad():
            for weight, row, col, original in reversed(originals):
                weight.data[row, col].copy_(original)


@contextmanager
def zero_coordinates(model, coordinates):
    interventions = {}
    for coordinate in coordinates:
        layer_name, row, col = normalize_coordinate(coordinate)
        interventions[(layer_name, row, col)] = 0.0
    with patch_weights(model, interventions):
        yield


def evaluate_set_ablation(
    model,
    tokenizer,
    texts,
    coordinates,
    batch_size=4,
    max_length=512,
    device="cuda",
    tail_fraction=0.05,
):
    """
    Zero a coordinate set and compare the edited model to the original model.
    """
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

            with zero_coordinates(model, coordinates):
                with torch.no_grad():
                    student_logits = logits_from_output(model(**kwargs))

            update_comparison_stats(stats, teacher_logits, student_logits, input_ids, attention_mask)
    finally:
        model.train(was_training)

    result = finalize_comparison_stats(stats, tail_fraction=tail_fraction)
    result["n_coordinates"] = len(coordinates)
    return result


def evaluate_coordinate_ablation(
    model,
    tokenizer,
    texts,
    layer_name,
    row,
    col,
    batch_size=4,
    max_length=512,
    device="cuda",
    tail_fraction=0.05,
):
    coordinate = (layer_name, row, col)
    result = evaluate_set_ablation(
        model,
        tokenizer,
        texts,
        [coordinate],
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        tail_fraction=tail_fraction,
    )
    result["coordinate"] = {"layer": layer_name, "row": int(row), "col": int(col)}
    return result


def evaluate_many_coordinates(
    model,
    tokenizer,
    texts,
    candidates,
    batch_size=4,
    max_length=512,
    device="cuda",
    tail_fraction=0.05,
):
    results = []
    for candidate in candidates:
        layer_name, row, col = normalize_coordinate(candidate)
        metrics = evaluate_coordinate_ablation(
            model,
            tokenizer,
            texts,
            layer_name,
            row,
            col,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            tail_fraction=tail_fraction,
        )
        item = dict(candidate) if isinstance(candidate, dict) else {
            "layer": layer_name,
            "row": int(row),
            "col": int(col),
        }
        item["ablation"] = metrics
        results.append(item)
    return results


def joint_ablation_interaction(
    model,
    tokenizer,
    texts,
    coordinates,
    batch_size=4,
    max_length=512,
    device="cuda",
    tail_fraction=0.05,
):
    """
    Compute I(S;P) = A_S - sum_c A_c using mean KL as A.
    """
    set_result = evaluate_set_ablation(
        model,
        tokenizer,
        texts,
        coordinates,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        tail_fraction=tail_fraction,
    )
    individual_results = []
    for coordinate in coordinates:
        layer_name, row, col = normalize_coordinate(coordinate)
        individual_results.append(
            evaluate_coordinate_ablation(
                model,
                tokenizer,
                texts,
                layer_name,
                row,
                col,
                batch_size=batch_size,
                max_length=max_length,
                device=device,
                tail_fraction=tail_fraction,
            )
        )

    individual_sum = sum(item["mean_kl"] for item in individual_results)
    return {
        "set": set_result,
        "individual": individual_results,
        "interaction": float(set_result["mean_kl"] - individual_sum),
        "individual_sum": float(individual_sum),
    }
