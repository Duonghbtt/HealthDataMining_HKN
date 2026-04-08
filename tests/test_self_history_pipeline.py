from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.models.fusion import FusionModule
from src.models.full_model import FullMedicationModel
from src.models.history_selector import SelfHistorySelector


def test_self_history_selector_masks_current_visit_and_padding() -> None:
    selector = SelfHistorySelector(hidden_dim=4, dropout=0.0, self_top_k=2)
    current_state = torch.tensor(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.5, 0.1, 0.2, 0.3],
        ],
        dtype=torch.float32,
    )
    state_sequence = torch.tensor(
        [
            [
                [0.9, 0.0, 0.0, 0.0],
                [0.7, 0.2, 0.0, 0.0],
                [0.4, 0.3, 0.2, 0.1],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.5, 0.1, 0.2, 0.3],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    visit_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    outputs = selector(
        current_state=current_state,
        state_sequence=state_sequence,
        visit_mask=visit_mask,
    )

    assert outputs["self_history_summary"].shape == (2, 4)
    assert outputs["self_attention_weights"].shape == (2, 4)
    assert outputs["selected_visit_indices"].shape == (2, 2)
    assert outputs["self_attention_weights"][0, 2].item() == 0.0
    assert outputs["self_attention_weights"][0, 3].item() == 0.0
    assert torch.equal(outputs["self_history_summary"][1], torch.zeros(4))
    assert torch.equal(outputs["selected_visit_indices"][1], torch.full((2,), -1, dtype=torch.long))


def test_fusion_module_emits_context_vector() -> None:
    module = FusionModule(hidden_dim=4, dropout=0.0)
    outputs = module(
        current_state=torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32),
        self_history_summary=torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32),
    )
    assert outputs["context_vector"].shape == (1, 4)
    assert torch.isfinite(outputs["context_vector"]).all()


def test_full_model_from_config_smoke() -> None:
    pytest.importorskip("pyarrow")
    from src.data.dataset import MIMICTrajectoryDataset, collate_batch

    dataset = MIMICTrajectoryDataset("train", "configs/data.yaml")
    if len(dataset) < 2:
        pytest.skip("Need at least two training records for the smoke test.")

    batch = collate_batch([dataset[0], dataset[1]])
    model = FullMedicationModel.from_config(
        data_config_path="configs/data.yaml",
        model_config_path="configs/model.yaml",
        train_config_path="configs/train.yaml",
    )
    outputs = model(batch)

    assert outputs["drug_logits"].shape == outputs["drug_probs"].shape
    assert outputs["context_vector"].shape[0] == batch["visit_mask"].shape[0]
    assert outputs["context_vector"].shape[1] == outputs["current_state"].shape[1]
    assert outputs["prediction_loss"] is not None
    assert outputs["ddi_loss"] is not None
    assert outputs["total_loss"] is not None
    assert torch.isfinite(outputs["drug_logits"]).all()
    assert torch.isfinite(outputs["drug_probs"]).all()
    assert torch.isfinite(outputs["context_vector"]).all()
