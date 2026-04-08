from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.ddi_regularization import DDIRegularizer


_VALID_REDUCTIONS = {"mean", "sum", "none"}


def _validate_reduction(reduction: str) -> str:
    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(f"reduction must be one of {_VALID_REDUCTIONS}, got {reduction!r}")
    return reduction


def _reduce_per_sample(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError(f"Expected per-sample tensor with shape (B,), got {tuple(values.shape)}")
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    return values


def extract_last_valid_targets(target_drugs: torch.Tensor, visit_mask: torch.Tensor) -> torch.Tensor:
    """Extract current-visit targets from [B, T, D] or passthrough [B, D]."""

    if not isinstance(target_drugs, torch.Tensor):
        raise TypeError(f"target_drugs must be a torch.Tensor, got {type(target_drugs)!r}")
    if target_drugs.ndim == 2:
        return target_drugs
    if target_drugs.ndim != 3:
        raise ValueError(f"target_drugs must have shape (B, D) or (B, T, D), got {tuple(target_drugs.shape)}")
    if not isinstance(visit_mask, torch.Tensor):
        raise TypeError(f"visit_mask must be a torch.Tensor, got {type(visit_mask)!r}")
    if visit_mask.ndim != 2:
        raise ValueError(f"visit_mask must have shape (B, T), got {tuple(visit_mask.shape)}")
    if tuple(target_drugs.shape[:2]) != tuple(visit_mask.shape):
        raise ValueError(
            "target_drugs and visit_mask must agree on batch/time dimensions: "
            f"got {tuple(target_drugs.shape[:2])} and {tuple(visit_mask.shape)}"
        )

    resolved_mask = visit_mask.to(device=target_drugs.device, dtype=torch.bool)
    valid_counts = resolved_mask.sum(dim=1)
    if bool((valid_counts <= 0).any().item()):
        raise ValueError("Each sample must contain at least one valid visit")

    last_indices = valid_counts.to(dtype=torch.long) - 1
    batch_indices = torch.arange(target_drugs.shape[0], device=target_drugs.device)
    return target_drugs[batch_indices, last_indices]


def _resolve_targets(
    target_drugs: torch.Tensor,
    visit_mask: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(target_drugs, torch.Tensor):
        raise TypeError(f"target_drugs must be a torch.Tensor, got {type(target_drugs)!r}")

    if target_drugs.ndim == 2:
        targets = target_drugs
    elif target_drugs.ndim == 3:
        if visit_mask is None:
            raise ValueError("visit_mask is required when target_drugs has shape (B, T, D)")
        targets = extract_last_valid_targets(target_drugs, visit_mask)
    else:
        raise ValueError(f"target_drugs must have shape (B, D) or (B, T, D), got {tuple(target_drugs.shape)}")

    targets = targets.to(device=device, dtype=dtype)
    if not torch.isfinite(targets).all():
        raise ValueError("target_drugs must contain only finite values")
    return targets


def _resolve_pos_weight(
    pos_weight: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    width: int,
) -> torch.Tensor | None:
    if pos_weight is None:
        return None
    resolved = torch.as_tensor(pos_weight, device=device, dtype=dtype)
    if resolved.ndim != 1 or resolved.shape[0] != width:
        raise ValueError(
            "pos_weight must have shape (D,) matching drug logits width: "
            f"got {tuple(resolved.shape)} for {width}"
        )
    return resolved


def compute_medication_losses(
    *,
    drug_logits: torch.Tensor,
    target_drugs: torch.Tensor,
    visit_mask: torch.Tensor | None = None,
    drug_probs: torch.Tensor | None = None,
    ddi_regularizer: DDIRegularizer | None = None,
    lambda_ddi: float = 0.0,
    pos_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> dict[str, Any]:
    """Compute BCE prediction loss, DDI loss, and total loss."""

    resolved_reduction = _validate_reduction(reduction)
    if not isinstance(drug_logits, torch.Tensor):
        raise TypeError(f"drug_logits must be a torch.Tensor, got {type(drug_logits)!r}")
    if drug_logits.ndim != 2:
        raise ValueError(f"drug_logits must have shape (B, D), got {tuple(drug_logits.shape)}")
    if not torch.isfinite(drug_logits).all():
        raise ValueError("drug_logits must contain only finite values")

    target_current = _resolve_targets(
        target_drugs,
        visit_mask,
        device=drug_logits.device,
        dtype=drug_logits.dtype,
    )
    if tuple(target_current.shape) != tuple(drug_logits.shape):
        raise ValueError(
            "Resolved targets must match drug_logits shape: "
            f"got {tuple(target_current.shape)} and {tuple(drug_logits.shape)}"
        )

    if drug_probs is None:
        resolved_probs = torch.sigmoid(drug_logits)
    else:
        resolved_probs = torch.as_tensor(drug_probs, device=drug_logits.device, dtype=drug_logits.dtype)
        if tuple(resolved_probs.shape) != tuple(drug_logits.shape):
            raise ValueError(
                "drug_probs must match drug_logits shape when provided: "
                f"got {tuple(resolved_probs.shape)} and {tuple(drug_logits.shape)}"
            )
        if not torch.isfinite(resolved_probs).all():
            raise ValueError("drug_probs must contain only finite values")

    resolved_pos_weight = _resolve_pos_weight(
        pos_weight,
        device=drug_logits.device,
        dtype=drug_logits.dtype,
        width=int(drug_logits.shape[1]),
    )

    prediction_loss_matrix = F.binary_cross_entropy_with_logits(
        drug_logits,
        target_current,
        pos_weight=resolved_pos_weight,
        reduction="none",
    )
    prediction_loss_per_sample = prediction_loss_matrix.mean(dim=1)
    prediction_loss = _reduce_per_sample(prediction_loss_per_sample, resolved_reduction)

    if ddi_regularizer is None:
        ddi_per_sample = torch.zeros(drug_logits.shape[0], device=drug_logits.device, dtype=drug_logits.dtype)
    else:
        ddi_per_sample = ddi_regularizer.compute_penalty_per_sample(resolved_probs)
        ddi_per_sample = ddi_per_sample.to(device=drug_logits.device, dtype=drug_logits.dtype)

    ddi_loss = _reduce_per_sample(ddi_per_sample, resolved_reduction)
    weighted_ddi_loss = ddi_loss * float(lambda_ddi)
    total_loss = prediction_loss + weighted_ddi_loss
    return {
        "prediction_loss": prediction_loss,
        "ddi_loss": ddi_loss,
        "weighted_ddi_loss": weighted_ddi_loss,
        "total_loss": total_loss,
        "target_current": target_current,
    }


class MedicationRecommendationLoss(nn.Module):
    """Compatibility wrapper around ``compute_medication_losses``."""

    def __init__(
        self,
        *,
        lambda_ddi: float = 0.0,
        ddi_regularizer: DDIRegularizer | None = None,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.lambda_ddi = float(lambda_ddi)
        self.ddi_regularizer = ddi_regularizer
        self.reduction = _validate_reduction(reduction)
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32))

    def forward(
        self,
        *,
        drug_logits: torch.Tensor,
        target_drugs: torch.Tensor,
        visit_mask: torch.Tensor | None = None,
        drug_probs: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        return compute_medication_losses(
            drug_logits=drug_logits,
            target_drugs=target_drugs,
            visit_mask=visit_mask,
            drug_probs=drug_probs,
            ddi_regularizer=self.ddi_regularizer,
            lambda_ddi=self.lambda_ddi,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )


__all__ = ["MedicationRecommendationLoss", "compute_medication_losses", "extract_last_valid_targets"]
