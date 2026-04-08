from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.models.medication_decoder import MedicationDecoder


@pytest.fixture
def fused_repr() -> torch.Tensor:
    return torch.tensor(
        [
            [0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2, 0.4],
            [-0.4, 0.6, 0.2, -0.3, 0.1, 0.7, -0.2, 0.0],
        ],
        dtype=torch.float32,
    )


def test_medication_decoder_forward_shapes_and_probabilities(fused_repr: torch.Tensor) -> None:
    decoder = MedicationDecoder(
        hidden_dim=8,
        drug_vocab_size=6,
        dropout=0.0,
    )

    outputs = decoder(context_vector=fused_repr)

    assert outputs["drug_logits"].shape == (2, 6)
    assert outputs["drug_probs"].shape == (2, 6)
    assert torch.isfinite(outputs["drug_logits"]).all()
    assert torch.isfinite(outputs["drug_probs"]).all()
    assert torch.all(outputs["drug_probs"] >= 0.0)
    assert torch.all(outputs["drug_probs"] <= 1.0)
