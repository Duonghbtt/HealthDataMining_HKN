from __future__ import annotations

from typing import Any

import torch
from torch import nn


class FusionModule(nn.Module):
    """Fuse current state and self-history summary into a single context vector."""

    def __init__(self, hidden_dim: int, *, dropout: float = 0.1, strategy: str = "gated") -> None:
        super().__init__()
        if str(strategy) != "gated":
            raise ValueError("FusionModule only supports the `gated` strategy in the new core pipeline.")

        self.hidden_dim = int(hidden_dim)
        self.candidate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        *,
        current_state: torch.Tensor,
        self_history_summary: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if current_state.ndim != 2:
            raise ValueError(f"current_state must have shape (B, H), got {tuple(current_state.shape)}")

        resolved_history = self_history_summary
        if resolved_history is None:
            resolved_history = torch.zeros_like(current_state)
        if tuple(resolved_history.shape) != tuple(current_state.shape):
            raise ValueError(
                "self_history_summary must match current_state shape: "
                f"got {tuple(resolved_history.shape)} and {tuple(current_state.shape)}"
            )

        joint = torch.cat([current_state, resolved_history], dim=-1)   # [B, 2H]
        candidate = self.candidate(joint)                               # [B, H]
        gate = self.gate(joint)                                         # [B, H]
        context_vector = self.norm(current_state + gate * candidate)    # [B, H]
        return {"context_vector": context_vector}


__all__ = ["FusionModule"]
