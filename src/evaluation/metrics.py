from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def _validate_binary_matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape (N, D), got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_same_shape(name_a: str, a: torch.Tensor, name_b: str, b: torch.Tensor) -> None:
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(
            f"{name_a} and {name_b} must share the same shape, got {tuple(a.shape)} and {tuple(b.shape)}"
        )


def _as_bool_matrix(value: torch.Tensor) -> torch.Tensor:
    return value.to(dtype=torch.bool)


def _to_numpy_matrix(name: str, value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (N, D), got {tuple(array.shape)}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _resolve_ddi_upper(ddi_matrix: torch.Tensor) -> torch.Tensor:
    matrix = _validate_binary_matrix("ddi_matrix", torch.as_tensor(ddi_matrix, dtype=torch.float32))
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"ddi_matrix must be square, got {tuple(matrix.shape)}")
    ddi_bool = (matrix > 0).to(dtype=torch.bool)
    ddi_bool = torch.logical_or(ddi_bool, ddi_bool.transpose(0, 1))
    ddi_bool.fill_diagonal_(False)
    return torch.triu(ddi_bool, diagonal=1)


def compute_samplewise_jaccard(
    y_true: torch.Tensor,
    y_pred_binary: torch.Tensor,
) -> torch.Tensor:
    """Compute sample-wise Jaccard scores for multi-label predictions."""

    true_tensor = _validate_binary_matrix("y_true", y_true)
    pred_tensor = _validate_binary_matrix("y_pred_binary", y_pred_binary)
    _validate_same_shape("y_true", true_tensor, "y_pred_binary", pred_tensor)

    true_mask = _as_bool_matrix(true_tensor)
    pred_mask = _as_bool_matrix(pred_tensor)
    intersection = torch.logical_and(true_mask, pred_mask).sum(dim=1, dtype=torch.float32)
    union = torch.logical_or(true_mask, pred_mask).sum(dim=1, dtype=torch.float32)
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def compute_samplewise_f1(
    y_true: torch.Tensor,
    y_pred_binary: torch.Tensor,
) -> torch.Tensor:
    """Compute sample-wise F1 scores for multi-label predictions."""

    true_tensor = _validate_binary_matrix("y_true", y_true)
    pred_tensor = _validate_binary_matrix("y_pred_binary", y_pred_binary)
    _validate_same_shape("y_true", true_tensor, "y_pred_binary", pred_tensor)

    true_mask = _as_bool_matrix(true_tensor)
    pred_mask = _as_bool_matrix(pred_tensor)
    true_positive = torch.logical_and(true_mask, pred_mask).sum(dim=1, dtype=torch.float32)
    false_positive = torch.logical_and(~true_mask, pred_mask).sum(dim=1, dtype=torch.float32)
    false_negative = torch.logical_and(true_mask, ~pred_mask).sum(dim=1, dtype=torch.float32)
    denominator = 2.0 * true_positive + false_positive + false_negative
    return torch.where(denominator > 0, (2.0 * true_positive) / denominator, torch.ones_like(denominator))


def binarize_predictions(drug_probs: torch.Tensor, threshold: float) -> torch.Tensor:
    """Threshold model probabilities into binary multi-label predictions."""

    probs = _validate_binary_matrix("drug_probs", drug_probs)
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")
    return probs >= float(threshold)


def multilabel_jaccard(y_true: torch.Tensor, y_pred_binary: torch.Tensor) -> float:
    """Sample-wise Jaccard averaged over the batch."""

    return float(compute_samplewise_jaccard(y_true, y_pred_binary).mean().item())


def multilabel_f1(y_true: torch.Tensor, y_pred_binary: torch.Tensor) -> float:
    """Sample-wise F1 averaged over the batch."""

    return float(compute_samplewise_f1(y_true, y_pred_binary).mean().item())


def compute_prauc(drug_probs: np.ndarray, y_true: np.ndarray) -> float:
    """Compute per-patient PRAUC and average over valid patients.

    This project defines PRAUC sample-wise: for each patient/sample, compute
    average precision over the drug vocabulary, skip samples with no positive
    labels, and return the mean score across valid patients.
    """

    probs = _to_numpy_matrix("drug_probs", drug_probs).astype(np.float64, copy=False)
    labels = _to_numpy_matrix("y_true", y_true).astype(np.float64, copy=False)
    if probs.shape != labels.shape:
        raise ValueError(f"drug_probs and y_true must share the same shape, got {probs.shape} and {labels.shape}")
    if labels.size > 0 and not np.logical_or(labels == 0.0, labels == 1.0).all():
        raise ValueError("y_true must be a binary matrix with values in {0, 1}")
    if probs.size > 0:
        assert probs.min() >= 0.0 and probs.max() <= 1.0, (
            "drug_probs must be in [0,1] - did you pass logits by mistake?"
        )

    per_sample_scores: list[float] = []
    for sample_index in range(labels.shape[0]):
        sample_true = labels[sample_index]
        if float(sample_true.sum()) <= 0.0:
            continue
        per_sample_scores.append(float(average_precision_score(sample_true, probs[sample_index])))

    if not per_sample_scores:
        return 0.0
    return float(sum(per_sample_scores) / float(len(per_sample_scores)))


def multilabel_prauc(y_true: torch.Tensor | np.ndarray, y_score: torch.Tensor | np.ndarray) -> float:
    """Compatibility wrapper for per-patient PRAUC computed from probabilities."""

    return compute_prauc(
        _to_numpy_matrix("drug_probs", y_score),
        _to_numpy_matrix("y_true", y_true),
    )


def compute_ddi_flags(y_pred_binary: torch.Tensor, ddi_matrix: torch.Tensor) -> torch.Tensor:
    """Return a boolean flag for whether each sample contains at least one DDI pair."""

    pred_tensor = _validate_binary_matrix("y_pred_binary", y_pred_binary)
    ddi_upper = _resolve_ddi_upper(ddi_matrix)
    if pred_tensor.shape[1] != ddi_upper.shape[0]:
        raise ValueError(
            "y_pred_binary width must match ddi_matrix width: "
            f"got {int(pred_tensor.shape[1])} and {int(ddi_upper.shape[0])}"
        )

    pred_mask = _as_bool_matrix(pred_tensor).to(device=ddi_upper.device)
    flags = torch.zeros(pred_mask.shape[0], dtype=torch.bool, device=ddi_upper.device)
    for sample_index in range(pred_mask.shape[0]):
        predicted_indices = torch.nonzero(pred_mask[sample_index], as_tuple=False).flatten()
        if predicted_indices.numel() < 2:
            continue
        sample_ddi = ddi_upper.index_select(0, predicted_indices).index_select(1, predicted_indices)
        flags[sample_index] = bool(sample_ddi.any().item())
    return flags


def compute_ddi_rate(y_pred_binary: torch.Tensor, ddi_matrix: torch.Tensor) -> dict[str, float]:
    """Compute dataset-level DDI rate from binary medication predictions."""

    pred_tensor = _validate_binary_matrix("y_pred_binary", y_pred_binary)
    ddi_upper = _resolve_ddi_upper(ddi_matrix)
    if pred_tensor.shape[1] != ddi_upper.shape[0]:
        raise ValueError(
            "y_pred_binary width must match ddi_matrix width: "
            f"got {int(pred_tensor.shape[1])} and {int(ddi_upper.shape[0])}"
        )

    pred_mask = _as_bool_matrix(pred_tensor).to(device=ddi_upper.device)
    total_predicted_pairs = 0.0
    total_interacting_pairs = 0.0
    patients_with_ddi = 0.0

    for sample_index in range(pred_mask.shape[0]):
        predicted_indices = torch.nonzero(pred_mask[sample_index], as_tuple=False).flatten()
        predicted_count = int(predicted_indices.numel())
        if predicted_count < 2:
            continue

        sample_total_pairs = float(predicted_count * (predicted_count - 1) // 2)
        sample_ddi = ddi_upper.index_select(0, predicted_indices).index_select(1, predicted_indices)
        sample_interacting_pairs = float(sample_ddi.sum(dtype=torch.float32).item())

        total_predicted_pairs += sample_total_pairs
        total_interacting_pairs += sample_interacting_pairs
        if sample_interacting_pairs > 0.0:
            patients_with_ddi += 1.0

    ddi_rate = 0.0 if total_predicted_pairs <= 0.0 else total_interacting_pairs / total_predicted_pairs
    return {
        "ddi_rate": float(ddi_rate),
        "total_predicted_pairs": float(total_predicted_pairs),
        "total_interacting_pairs": float(total_interacting_pairs),
        "patients_with_ddi": float(patients_with_ddi),
        "num_samples": float(pred_tensor.shape[0]),
    }


def compute_core_metrics(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    threshold: float,
    ddi_matrix: torch.Tensor,
) -> dict[str, float]:
    """Compute the core evaluation metric bundle for medication recommendation."""

    y_pred_binary = binarize_predictions(y_score, threshold)
    ddi_summary = compute_ddi_rate(y_pred_binary, ddi_matrix)
    return {
        "jaccard": multilabel_jaccard(y_true, y_pred_binary),
        "f1": multilabel_f1(y_true, y_pred_binary),
        "prauc": compute_prauc(
            _to_numpy_matrix("drug_probs", y_score),
            _to_numpy_matrix("y_true", y_true),
        ),
        **ddi_summary,
    }


__all__ = [
    "binarize_predictions",
    "compute_core_metrics",
    "compute_ddi_flags",
    "compute_ddi_rate",
    "compute_prauc",
    "compute_samplewise_f1",
    "compute_samplewise_jaccard",
    "multilabel_f1",
    "multilabel_jaccard",
    "multilabel_prauc",
]
