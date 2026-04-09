from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

_VALID_REDUCTIONS = {"mean", "sum", "none"}
_LAMBDA_DDI_DISABLED_WARNING_SHOWN = False


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


def _build_pos_weight_from_avg_pos(
    *,
    vocab_size: int,
    avg_pos: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")

    resolved_avg_pos = float(avg_pos)
    if resolved_avg_pos <= 0.0:
        raise ValueError(f"avg_pos must be > 0 to build pos_weight, got {resolved_avg_pos}")
    if resolved_avg_pos >= float(vocab_size):
        raise ValueError(
            "avg_pos must be smaller than vocab_size to build a positive pos_weight: "
            f"got avg_pos={resolved_avg_pos} and vocab_size={vocab_size}"
        )

    pos_weight_scalar = (float(vocab_size) - resolved_avg_pos) / resolved_avg_pos
    return torch.full((vocab_size,), pos_weight_scalar, device=device, dtype=dtype)


def _resolve_pos_weight(
    pos_weight: torch.Tensor | None,
    avg_pos: float | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    width: int,
) -> torch.Tensor | None:
    """Resolve class imbalance weighting from either an explicit tensor or avg_pos."""

    if pos_weight is not None and avg_pos is not None:
        raise ValueError("Provide either explicit pos_weight or avg_pos, but not both")
    if pos_weight is None:
        if avg_pos is None:
            return None
        return _build_pos_weight_from_avg_pos(
            vocab_size=width,
            avg_pos=avg_pos,
            device=device,
            dtype=dtype,
        )

    resolved = torch.as_tensor(pos_weight, device=device, dtype=dtype)
    if resolved.ndim != 1 or resolved.shape[0] != width:
        raise ValueError(
            "pos_weight must have shape (D,) matching drug logits width: "
            f"got {tuple(resolved.shape)} for {width}"
        )
    if not torch.isfinite(resolved).all():
        raise ValueError("pos_weight must contain only finite values")
    return resolved


def _warn_if_ddi_disabled(lambda_ddi: float) -> None:
    global _LAMBDA_DDI_DISABLED_WARNING_SHOWN
    if float(lambda_ddi) == 0.0 and not _LAMBDA_DDI_DISABLED_WARNING_SHOWN:
        print("WARNING: lambda_ddi=0, DDI loss is disabled. DDI rate will not be optimized.")
        _LAMBDA_DDI_DISABLED_WARNING_SHOWN = True


def _resolve_ddi_matrix(
    ddi_matrix: torch.Tensor | None,
    *,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if ddi_matrix is None:
        return None

    resolved = torch.as_tensor(ddi_matrix, device=device, dtype=dtype)
    if resolved.ndim != 2:
        raise ValueError(f"ddi_matrix must have shape (D, D), got {tuple(resolved.shape)}")
    if resolved.shape[0] != resolved.shape[1]:
        raise ValueError(f"ddi_matrix must be square, got {tuple(resolved.shape)}")
    if int(resolved.shape[0]) != int(width):
        raise ValueError(
            "ddi_matrix width must match drug logits width: "
            f"got {int(resolved.shape[0])} and {int(width)}"
        )
    if not torch.isfinite(resolved).all():
        raise ValueError("ddi_matrix must contain only finite values")
    return resolved


def _validate_optional_drug_probs(
    drug_probs: torch.Tensor,
    *,
    drug_logits: torch.Tensor,
    expected_probs: torch.Tensor,
) -> None:
    if tuple(drug_probs.shape) != tuple(drug_logits.shape):
        raise ValueError(
            "drug_probs must match drug_logits shape when provided: "
            f"got {tuple(drug_probs.shape)} and {tuple(drug_logits.shape)}"
        )
    if not torch.isfinite(drug_probs).all():
        raise ValueError("drug_probs must contain only finite values")
    if bool(((drug_probs < 0.0) | (drug_probs > 1.0)).any().item()):
        raise ValueError("drug_probs must contain values in [0, 1]")
    if not torch.allclose(drug_probs, expected_probs, rtol=1e-4, atol=1e-6):
        raise ValueError(
            "drug_probs must match torch.sigmoid(drug_logits); external probabilities are not authoritative"
        )


def compute_medication_losses(
    *,
    drug_logits: torch.Tensor,
    target_drugs: torch.Tensor,
    visit_mask: torch.Tensor | None = None,
    drug_probs: torch.Tensor | None = None,
    ddi_matrix: torch.Tensor | None = None,
    lambda_ddi: float = 0.0,
    pos_weight: torch.Tensor | None = None,
    avg_pos: float | None = None,
    reduction: str = "mean",
    training: bool = True,
) -> dict[str, Any]:
    """Compute medication losses from raw logits.

    Class imbalance weighting can be supplied either via an explicit ``pos_weight``
    tensor with shape ``(D,)`` or via ``avg_pos`` so the function auto-builds a
    uniform per-class weight using the current medication vocabulary width.
    """

    resolved_reduction = _validate_reduction(reduction)
    if not isinstance(drug_logits, torch.Tensor):
        raise TypeError(f"drug_logits must be a torch.Tensor, got {type(drug_logits)!r}")
    if drug_logits.ndim != 2:
        raise ValueError(f"drug_logits must have shape (B, D), got {tuple(drug_logits.shape)}")
    if not torch.isfinite(drug_logits).all():
        raise ValueError("drug_logits must contain only finite values")
    training = bool(training and torch.is_grad_enabled())
    assert drug_logits.requires_grad or not training, (
        "drug_logits must be raw logits, not sigmoid output"
    )
    _warn_if_ddi_disabled(lambda_ddi)

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

    resolved_probs = torch.sigmoid(drug_logits)
    if drug_probs is not None:
        provided_probs = torch.as_tensor(drug_probs, device=drug_logits.device, dtype=drug_logits.dtype)
        _validate_optional_drug_probs(
            provided_probs,
            drug_logits=drug_logits,
            expected_probs=resolved_probs,
        )

    resolved_pos_weight = _resolve_pos_weight(
        pos_weight,
        avg_pos,
        device=drug_logits.device,
        dtype=drug_logits.dtype,
        width=int(drug_logits.shape[1]),
    )
    resolved_ddi_matrix = _resolve_ddi_matrix(
        ddi_matrix,
        width=int(drug_logits.shape[1]),
        device=resolved_probs.device,
        dtype=resolved_probs.dtype,
    )

    # Multi-label BCE prediction loss is computed directly on raw pre-sigmoid logits.
    prediction_loss_matrix = F.binary_cross_entropy_with_logits(
        drug_logits,
        target_current,
        pos_weight=resolved_pos_weight,
        reduction="none",
    )
    prediction_loss_per_sample = prediction_loss_matrix.mean(dim=1)
    prediction_loss = _reduce_per_sample(prediction_loss_per_sample, resolved_reduction)

    # DDI loss computes expected co-medication pair strength under predicted probabilities
    # and penalizes predicted pairs that are known to interact in the DDI matrix.
    if resolved_ddi_matrix is None:
        ddi_per_sample = torch.zeros(drug_logits.shape[0], device=drug_logits.device, dtype=drug_logits.dtype)
    else:
        pred_pairs = resolved_probs.unsqueeze(2) * resolved_probs.unsqueeze(1)
        ddi_per_sample = (pred_pairs * resolved_ddi_matrix.unsqueeze(0)).sum(dim=(1, 2))

    ddi_loss = _reduce_per_sample(ddi_per_sample, resolved_reduction)
    weighted_ddi_loss = ddi_loss * float(lambda_ddi)
    # Total loss combines prediction quality and DDI regularization.
    total_loss = prediction_loss + float(lambda_ddi) * ddi_loss
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
        ddi_matrix: torch.Tensor | None = None,
        pos_weight: torch.Tensor | None = None,
        avg_pos: float | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.lambda_ddi = float(lambda_ddi)
        self.reduction = _validate_reduction(reduction)
        if pos_weight is not None and avg_pos is not None:
            raise ValueError("Provide either explicit pos_weight or avg_pos, but not both")
        self.avg_pos = None if avg_pos is None else float(avg_pos)
        if ddi_matrix is None:
            self.register_buffer("ddi_matrix", None)
        else:
            self.register_buffer("ddi_matrix", torch.as_tensor(ddi_matrix, dtype=torch.float32))
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
            ddi_matrix=self.ddi_matrix,
            lambda_ddi=self.lambda_ddi,
            pos_weight=self.pos_weight,
            avg_pos=self.avg_pos,
            reduction=self.reduction,
            training=self.training,
        )


__all__ = ["MedicationRecommendationLoss", "compute_medication_losses", "extract_last_valid_targets"]
