from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.features.diagnosis_encoder import DiagnosisEncoder
from src.features.procedure_encoder import ProcedureEncoder


def _require_tensor(batch: Mapping[str, Any], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"Batch is missing tensor field `{key}`.")
    return value


def _optional_tensor(batch: Mapping[str, Any], key: str) -> torch.Tensor | None:
    value = batch.get(key)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Optional batch field `{key}` must be a torch.Tensor when provided.")
    return value


def _masked_average(
    embedding: nn.Embedding,
    indices: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    vectors = embedding(indices)
    weights = mask.to(dtype=vectors.dtype).unsqueeze(-1)
    summed = (vectors * weights).sum(dim=-2)
    denom = weights.sum(dim=-2).clamp(min=1.0)
    return summed / denom


class _NumericProjector(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.projection = None if self.in_features <= 0 else nn.Linear(self.in_features, self.out_features)

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(f"Numeric values must have shape (B, T, F), got {tuple(values.shape)}")
        if values.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected numeric feature width {self.in_features}, got {int(values.shape[-1])}"
            )
        if self.projection is None:
            return values.new_zeros(values.shape[0], values.shape[1], self.out_features)

        masked_values = values
        if mask is not None:
            if tuple(mask.shape) != tuple(values.shape):
                raise ValueError(
                    f"Numeric mask shape {tuple(mask.shape)} must match values shape {tuple(values.shape)}"
                )
            masked_values = values * mask.to(dtype=values.dtype)
        return self.projection(masked_values)


class PatientStateEncoder(nn.Module):
    def __init__(
        self,
        diagnosis_vocab_size: int,
        procedure_vocab_size: int,
        drug_vocab_size: int,
        num_lab_features: int,
        num_vital_features: int,
        *,
        code_embedding_dim: int = 64,
        medication_embedding_dim: int = 64,
        numeric_projection_dim: int = 32,
        time_embedding_dim: int = 32,
        visit_hidden_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.visit_hidden_dim = int(visit_hidden_dim)

        self.diagnosis_encoder = DiagnosisEncoder(diagnosis_vocab_size, code_embedding_dim, padding_idx=0)
        self.procedure_encoder = ProcedureEncoder(procedure_vocab_size, code_embedding_dim, padding_idx=0)
        self.medication_embedding = nn.Embedding(drug_vocab_size, medication_embedding_dim, padding_idx=0)
        self.lab_projection = _NumericProjector(num_lab_features, numeric_projection_dim)
        self.vital_projection = _NumericProjector(num_vital_features, numeric_projection_dim)
        self.time_projection = nn.Linear(1, time_embedding_dim)

        fused_dim = (
            code_embedding_dim
            + code_embedding_dim
            + medication_embedding_dim
            + numeric_projection_dim
            + numeric_projection_dim
            + time_embedding_dim
        )
        self.visit_projection = nn.Sequential(
            nn.Linear(fused_dim, visit_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(visit_hidden_dim, hidden_dim, batch_first=True)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        diag_codes = _require_tensor(batch, "diag_codes")
        proc_codes = _require_tensor(batch, "proc_codes")
        lab_values = _require_tensor(batch, "lab_values")
        vital_values = _require_tensor(batch, "vital_values")
        med_history = _require_tensor(batch, "med_history")
        visit_mask = _require_tensor(batch, "visit_mask").to(dtype=torch.bool)
        time_delta_hours = _optional_tensor(batch, "time_delta_hours")

        if visit_mask.ndim != 2:
            raise ValueError(f"visit_mask must have shape (B, T), got {tuple(visit_mask.shape)}")
        if diag_codes.shape[:2] != visit_mask.shape:
            raise ValueError("diag_codes must align with visit_mask on batch/time dimensions")
        if proc_codes.shape[:2] != visit_mask.shape:
            raise ValueError("proc_codes must align with visit_mask on batch/time dimensions")
        if lab_values.shape[:2] != visit_mask.shape:
            raise ValueError("lab_values must align with visit_mask on batch/time dimensions")
        if vital_values.shape[:2] != visit_mask.shape:
            raise ValueError("vital_values must align with visit_mask on batch/time dimensions")
        if med_history.shape[:2] != visit_mask.shape:
            raise ValueError("med_history must align with visit_mask on batch/time dimensions")

        if time_delta_hours is None:
            time_delta_hours = torch.zeros(
                visit_mask.shape[0],
                visit_mask.shape[1],
                dtype=torch.float32,
                device=visit_mask.device,
            )
        if tuple(time_delta_hours.shape) != tuple(visit_mask.shape):
            raise ValueError(
                "time_delta_hours must align with visit_mask on batch/time dimensions: "
                f"got {tuple(time_delta_hours.shape)} and {tuple(visit_mask.shape)}"
            )

        diag_mask = _optional_tensor(batch, "diag_mask")
        proc_mask = _optional_tensor(batch, "proc_mask")
        med_history_mask = _optional_tensor(batch, "med_history_mask")
        lab_mask = _optional_tensor(batch, "lab_mask")
        vital_mask = _optional_tensor(batch, "vital_mask")

        resolved_diag_mask = (
            diag_mask.to(device=diag_codes.device, dtype=torch.bool)
            if isinstance(diag_mask, torch.Tensor)
            else diag_codes.ne(0)
        )
        resolved_proc_mask = (
            proc_mask.to(device=proc_codes.device, dtype=torch.bool)
            if isinstance(proc_mask, torch.Tensor)
            else proc_codes.ne(0)
        )
        resolved_history_mask = (
            med_history_mask.to(device=med_history.device, dtype=torch.bool)
            if isinstance(med_history_mask, torch.Tensor)
            else med_history.ne(0)
        )
        resolved_lab_mask = (
            lab_mask.to(device=lab_values.device, dtype=torch.bool)
            if isinstance(lab_mask, torch.Tensor)
            else torch.ones_like(lab_values, dtype=torch.bool)
        )
        resolved_vital_mask = (
            vital_mask.to(device=vital_values.device, dtype=torch.bool)
            if isinstance(vital_mask, torch.Tensor)
            else torch.ones_like(vital_values, dtype=torch.bool)
        )

        # diag_repr/proc_repr/med_repr/lab_repr/vital_repr/time_repr: [B, T, *]
        diag_repr = self.diagnosis_encoder(diag_codes, resolved_diag_mask)
        proc_repr = self.procedure_encoder(proc_codes, resolved_proc_mask)
        med_repr = _masked_average(self.medication_embedding, med_history, resolved_history_mask)
        lab_repr = self.lab_projection(lab_values, resolved_lab_mask)
        vital_repr = self.vital_projection(vital_values, resolved_vital_mask)
        time_repr = self.time_projection(torch.log1p(time_delta_hours).unsqueeze(-1))

        # visit_repr: [B, T, V]
        visit_repr = self.visit_projection(
            torch.cat([diag_repr, proc_repr, med_repr, lab_repr, vital_repr, time_repr], dim=-1)
        )

        valid_lengths = visit_mask.sum(dim=-1).clamp(min=1).to(dtype=torch.long)
        packed = pack_padded_sequence(
            visit_repr,
            valid_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, hidden = self.gru(packed)
        state_sequence, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=visit_repr.shape[1],
        )
        pooled_state = hidden[-1]

        if tuple(state_sequence.shape[:2]) != tuple(visit_mask.shape):
            raise AssertionError("state_sequence batch/time shape must match visit_mask")
        if state_sequence.shape[-1] != self.hidden_dim:
            raise AssertionError("state_sequence hidden width does not match encoder hidden_dim")
        if visit_repr.shape[-1] != self.visit_hidden_dim:
            raise AssertionError("visit_repr width does not match visit_hidden_dim")

        return {
            "visit_repr": visit_repr,          # [B, T, V]
            "state_sequence": state_sequence,  # [B, T, H]
            "pooled_state": pooled_state,      # [B, H]
            "visit_mask": visit_mask,          # [B, T]
        }
