"""
Mechanistic interpretability tools for scalar weight-handle experiments.
"""

from .ablation import (
    evaluate_coordinate_ablation,
    evaluate_many_coordinates,
    evaluate_set_ablation,
    joint_ablation_interaction,
    patch_weights,
    zero_coordinates,
)
from .candidates import (
    functional_under_outlier_candidates,
    known_super_weight_candidates,
    matched_control_candidates,
    top_magnitude_candidates,
)
from .metrics import (
    compute_kl_divergence,
    compute_perplexity,
    compute_teacher_student_metrics,
)
from .patching import patch_back_sufficiency, rank_one_patch_sufficiency, spreadability_tests
from .residual import collect_residual_streams, track_residual_perturbation
from .taxonomy import classify_candidate, classify_results

__all__ = [
    "classify_candidate",
    "classify_results",
    "collect_residual_streams",
    "compute_kl_divergence",
    "compute_perplexity",
    "compute_teacher_student_metrics",
    "evaluate_coordinate_ablation",
    "evaluate_many_coordinates",
    "evaluate_set_ablation",
    "functional_under_outlier_candidates",
    "joint_ablation_interaction",
    "known_super_weight_candidates",
    "matched_control_candidates",
    "patch_back_sufficiency",
    "patch_weights",
    "rank_one_patch_sufficiency",
    "spreadability_tests",
    "top_magnitude_candidates",
    "track_residual_perturbation",
    "zero_coordinates",
]
