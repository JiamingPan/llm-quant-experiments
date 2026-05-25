"""
Rule-based taxonomy assignment from intervention results.
"""


def classify_candidate(result, thresholds=None):
    thresholds = thresholds or {}
    high_kl = thresholds.get("super_weight_mean_kl", 0.02)
    low_kl = thresholds.get("under_outlier_mean_kl", 1e-4)
    high_tail = thresholds.get("tail_cvar_kl", 0.1)
    stable = thresholds.get("stability", 0.5)
    interaction = thresholds.get("shadow_interaction", 0.01)

    ablation = result.get("ablation", result)
    mean_kl = float(ablation.get("mean_kl", 0.0))
    tail_kl = float(ablation.get("tail_cvar_kl", 0.0))
    flip_rate = float(ablation.get("top_token_flip_rate", 0.0))
    stability_score = float(result.get("residual", {}).get("max_stability", 0.0))
    joint = float(result.get("joint_interaction", 0.0))

    if mean_kl >= high_kl or tail_kl >= high_tail or flip_rate >= thresholds.get("super_weight_flip_rate", 0.05):
        return "Super Weight"
    if joint >= interaction:
        return "Shadow Outlier"
    if mean_kl <= low_kl and stability_score >= stable:
        return "Silent Stabilizer"
    if mean_kl <= low_kl:
        return "Under-Outlier"
    if stability_score >= stable:
        return "Obsolete Anchor"
    return "Unresolved"


def classify_results(results, thresholds=None):
    classified = []
    for item in results:
        row = dict(item)
        row["taxonomy"] = classify_candidate(item, thresholds=thresholds)
        classified.append(row)
    return classified


def taxonomy_counts(classified):
    counts = {}
    for item in classified:
        label = item.get("taxonomy", "Unresolved")
        counts[label] = counts.get(label, 0) + 1
    return counts
