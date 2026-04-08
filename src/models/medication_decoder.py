from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _validate_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


class MedicationDecoder(nn.Module):
    """Decode a fused context vector into medication logits and probabilities."""

    def __init__(
        self,
        hidden_dim: int,
        drug_vocab_size: int,
        *,
        dropout: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.hidden_dim = _validate_positive_int("hidden_dim", int(hidden_dim))
        self.drug_vocab_size = _validate_positive_int("drug_vocab_size", int(drug_vocab_size))
        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout!r}")

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.drug_vocab_size),
        )

    def forward(self, context_vector: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        if not isinstance(context_vector, torch.Tensor):
            raise TypeError(f"context_vector must be a torch.Tensor, got {type(context_vector)!r}")
        if context_vector.ndim != 2:
            raise ValueError(f"context_vector must have shape (B, H), got {tuple(context_vector.shape)}")
        if context_vector.shape[1] != self.hidden_dim:
            raise ValueError(
                "context_vector hidden dimension mismatch: "
                f"expected {self.hidden_dim}, got {int(context_vector.shape[1])}"
            )

        drug_logits = self.decoder(context_vector)  # [B, D]
        drug_probs = torch.sigmoid(drug_logits)     # [B, D]
        return {
            "drug_logits": drug_logits,
            "drug_probs": drug_probs,
        }


__all__ = ["MedicationDecoder"]
