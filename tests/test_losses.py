from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.models.ddi_regularization import DDIRegularizer
from src.training.losses import compute_medication_losses, extract_last_valid_targets


def test_extract_last_valid_targets_uses_last_valid_visit() -> None:
    target_drugs = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [9.0, 9.0, 9.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    visit_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.bool,
    )

    target_current = extract_last_valid_targets(target_drugs, visit_mask)

    assert torch.equal(target_current[0], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(target_current[1], torch.tensor([0.0, 0.0, 0.0]))


def test_compute_medication_losses_returns_all_components() -> None:
    ddi_adj = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    logits = torch.tensor(
        [
            [0.5, -0.2, 0.1],
            [-0.4, 0.7, 0.3],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    visit_mask = torch.tensor(
        [
            [1, 1],
            [1, 1],
        ],
        dtype=torch.bool,
    )

    loss_outputs = compute_medication_losses(
        drug_logits=logits,
        target_drugs=targets,
        visit_mask=visit_mask,
        ddi_regularizer=DDIRegularizer(ddi_adj, reduction="mean"),
        lambda_ddi=0.05,
    )

    assert loss_outputs["target_current"].shape == (2, 3)
    assert torch.isfinite(loss_outputs["prediction_loss"])
    assert torch.isfinite(loss_outputs["ddi_loss"])
    assert torch.isfinite(loss_outputs["weighted_ddi_loss"])
    assert torch.isfinite(loss_outputs["total_loss"])
    assert torch.allclose(
        loss_outputs["total_loss"],
        loss_outputs["prediction_loss"] + loss_outputs["weighted_ddi_loss"],
    )
