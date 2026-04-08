from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _validate_shapes(
    current_state: torch.Tensor,
    state_sequence: torch.Tensor,
    visit_mask: torch.Tensor,
    hidden_dim: int,
) -> None:
    if current_state.ndim != 2:
        raise ValueError(f"current_state must have shape (B, H), got {tuple(current_state.shape)}")
    if state_sequence.ndim != 3:
        raise ValueError(f"state_sequence must have shape (B, T, H), got {tuple(state_sequence.shape)}")
    if visit_mask.ndim != 2:
        raise ValueError(f"visit_mask must have shape (B, T), got {tuple(visit_mask.shape)}")
    if tuple(state_sequence.shape[:2]) != tuple(visit_mask.shape):
        raise ValueError(
            "state_sequence and visit_mask must align on batch/time dimensions: "
            f"got {tuple(state_sequence.shape[:2])} and {tuple(visit_mask.shape)}"
        )
    if current_state.shape[0] != state_sequence.shape[0]:
        raise ValueError(
            "current_state and state_sequence must align on batch dimension: "
            f"got {tuple(current_state.shape)} and {tuple(state_sequence.shape)}"
        )
    if current_state.shape[1] != state_sequence.shape[2]:
        raise ValueError(
            "current_state and state_sequence must align on hidden dimension: "
            f"got {tuple(current_state.shape)} and {tuple(state_sequence.shape)}"
        )
    if current_state.shape[1] != hidden_dim:
        raise ValueError(f"Expected hidden dimension {hidden_dim}, got {int(current_state.shape[1])}")


def _extract_selected_visit_indices(
    attention_weights: torch.Tensor,
    history_mask: torch.Tensor,
    *,
    max_selected_visits: int | None,
) -> torch.Tensor:
    batch_size, time_steps = attention_weights.shape
    device = attention_weights.device
    if max_selected_visits is None:
        selection_width = time_steps
    else:
        if int(max_selected_visits) <= 0:
            raise ValueError("max_selected_visits must be positive when provided")
        selection_width = min(int(max_selected_visits), time_steps)

    if selection_width == 0:
        return torch.empty(batch_size, 0, dtype=torch.long, device=device)

    masked_weights = attention_weights.masked_fill(~history_mask, -1.0)
    ordered_indices = torch.argsort(masked_weights, dim=-1, descending=True)
    selected_indices = ordered_indices[:, :selection_width].to(dtype=torch.long)
    selected_mask = history_mask.gather(1, selected_indices)
    return torch.where(selected_mask, selected_indices, torch.full_like(selected_indices, -1))


class SelfHistorySelector(nn.Module):
    """Attend only over previous visits from the same patient."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        dropout: float = 0.1,
        max_selected_visits: int | None = None,
        self_top_k: int | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_selected_visits = self_top_k if self_top_k is not None else max_selected_visits
        if self.max_selected_visits is not None and int(self.max_selected_visits) <= 0:
            raise ValueError("max_selected_visits must be positive when provided")

        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        current_state: torch.Tensor,
        state_sequence: torch.Tensor,
        visit_mask: torch.Tensor,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        _validate_shapes(current_state, state_sequence, visit_mask, self.hidden_dim)

        resolved_mask = visit_mask.to(device=state_sequence.device, dtype=torch.bool)
        valid_counts = resolved_mask.sum(dim=1)
        if bool((valid_counts <= 0).any().item()):
            raise ValueError("Each sample must contain at least one valid visit")

        batch_size, time_steps, _ = state_sequence.shape
        visit_indices = torch.arange(time_steps, device=state_sequence.device).unsqueeze(0).expand(batch_size, -1)
        last_valid_index = valid_counts.to(dtype=torch.long) - 1
        history_mask = resolved_mask & (visit_indices < last_valid_index.unsqueeze(-1))

        query = self.query_proj(current_state)                         # [B, H]
        keys = self.key_proj(state_sequence)                           # [B, T, H]
        values = self.value_proj(state_sequence)                       # [B, T, H]
        scores = torch.einsum("bth,bh->bt", keys, query) / (self.hidden_dim ** 0.5)  # [B, T]
        scores = scores.masked_fill(~history_mask, -1.0e9)

        attention_weights = torch.softmax(scores, dim=-1)             # [B, T]
        attention_weights = torch.where(history_mask, attention_weights, torch.zeros_like(attention_weights))
        normalizer = attention_weights.sum(dim=-1, keepdim=True)
        attention_weights = attention_weights / torch.where(normalizer > 0, normalizer, torch.ones_like(normalizer))

        if self.max_selected_visits is not None:
            sparse_rows: list[torch.Tensor] = []
            for row_index in range(batch_size):
                row_mask = history_mask[row_index]
                row_sparse = torch.zeros_like(attention_weights[row_index])
                valid_count = int(row_mask.sum().item())
                if valid_count > 0:
                    keep = min(int(self.max_selected_visits), valid_count)
                    row_scores = attention_weights[row_index].masked_fill(~row_mask, -1.0)
                    ordered_positions = torch.argsort(row_scores, descending=True)
                    selected_positions = ordered_positions[:keep]
                    row_sparse.scatter_(0, selected_positions, attention_weights[row_index].index_select(0, selected_positions))
                    row_sparse = row_sparse / row_sparse.sum().clamp(min=torch.finfo(row_sparse.dtype).eps)
                sparse_rows.append(row_sparse)
            attention_weights = torch.stack(sparse_rows, dim=0)

        self_history_summary = (values * attention_weights.unsqueeze(-1)).sum(dim=1)  # [B, H]
        has_history = history_mask.any(dim=1, keepdim=True)
        self_history_summary = torch.where(has_history, self_history_summary, torch.zeros_like(self_history_summary))

        selected_visit_indices = _extract_selected_visit_indices(
            attention_weights,
            history_mask,
            max_selected_visits=None if self.max_selected_visits is None else int(self.max_selected_visits),
        )
        return {
            "self_history_summary": self_history_summary,
            "self_attention_weights": attention_weights,
            "selected_visit_indices": selected_visit_indices,
        }


HistorySelector = SelfHistorySelector

__all__ = ["HistorySelector", "SelfHistorySelector"]

