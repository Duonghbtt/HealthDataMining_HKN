from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from src.data.build_vocab import load_vocab_bundle
from src.data.dataset import MIMICTrajectoryDataset
from src.models.ddi_regularization import load_ddi_matrix
from src.models.fusion import FusionModule
from src.models.history_selector import HistorySelector, SelfHistorySelector
from src.models.medication_decoder import MedicationDecoder
from src.models.patient_state_encoder import PatientStateEncoder
from src.training.losses import compute_medication_losses, extract_last_valid_targets
from src.utils.io import load_yaml_config, resolve_path


def extract_last_valid_state(state_sequence: torch.Tensor, visit_mask: torch.Tensor) -> torch.Tensor:
    """Extract the hidden state of the last valid visit.

    Shapes:
    - state_sequence: [B, T, H]
    - visit_mask: [B, T]
    - return: [B, H]
    """

    if state_sequence.ndim != 3:
        raise ValueError(f"state_sequence must have shape (B, T, H), got {tuple(state_sequence.shape)}")
    if visit_mask.ndim != 2:
        raise ValueError(f"visit_mask must have shape (B, T), got {tuple(visit_mask.shape)}")
    if tuple(state_sequence.shape[:2]) != tuple(visit_mask.shape):
        raise ValueError(
            "state_sequence and visit_mask must align on batch/time dimensions: "
            f"got {tuple(state_sequence.shape[:2])} and {tuple(visit_mask.shape)}"
        )

    resolved_mask = visit_mask.to(device=state_sequence.device, dtype=torch.bool)
    valid_counts = resolved_mask.sum(dim=1)
    if bool((valid_counts <= 0).any().item()):
        raise ValueError("Each sample must contain at least one valid visit")

    last_indices = valid_counts.to(dtype=torch.long) - 1
    batch_indices = torch.arange(state_sequence.shape[0], device=state_sequence.device)
    return state_sequence[batch_indices, last_indices]


def _infer_numeric_feature_sizes(data_config_path: str | Path) -> tuple[int, int]:
    dataset = MIMICTrajectoryDataset("train", data_config_path)
    lab_feature_size = int(getattr(dataset, "default_lab_feature_size", 0))
    vital_feature_size = int(getattr(dataset, "default_vital_feature_size", 0))
    if lab_feature_size > 0 or vital_feature_size > 0:
        return lab_feature_size, vital_feature_size

    if len(dataset) <= 0:
        raise ValueError("Training split is empty; cannot infer lab/vital feature sizes from dataset.")
    sample = dataset[0]
    return int(sample.get("lab_feature_size", 0)), int(sample.get("vital_feature_size", 0))


def _load_optional_ddi_matrix(
    train_config_path: str | Path | None,
) -> tuple[torch.Tensor | None, float]:
    if train_config_path is None:
        return None, 0.0

    train_config = load_yaml_config(train_config_path)
    lambda_ddi = float(train_config.get("loss", {}).get("lambda_ddi", 0.0))
    ddi_path_value = train_config.get("paths", {}).get("ddi_matrix_path")
    if not ddi_path_value:
        return None, lambda_ddi

    ddi_path = resolve_path(train_config["_project_root"], ddi_path_value)
    if not ddi_path.exists():
        return None, lambda_ddi
    return load_ddi_matrix(ddi_path, device="cpu"), lambda_ddi


class FullMedicationModel(nn.Module):
    """Self-history-only medication recommendation model."""

    def __init__(
        self,
        encoder: PatientStateEncoder,
        history_selector: SelfHistorySelector | HistorySelector,
        fusion_module: FusionModule,
        *,
        medication_decoder: MedicationDecoder | None = None,
        decoder: MedicationDecoder | None = None,
        ddi_matrix: torch.Tensor | None = None,
        lambda_ddi: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.self_history_selector = history_selector
        self.fusion_module = fusion_module
        self.medication_decoder = medication_decoder if medication_decoder is not None else decoder
        self.lambda_ddi = float(lambda_ddi)
        if ddi_matrix is None:
            self.register_buffer("ddi_matrix", None)
        else:
            self.register_buffer("ddi_matrix", torch.as_tensor(ddi_matrix, dtype=torch.float32))

    @classmethod
    def from_config(
        cls,
        *,
        data_config_path: str | Path,
        model_config_path: str | Path,
        train_config_path: str | Path | None = None,
    ) -> "FullMedicationModel":
        data_config = load_yaml_config(data_config_path)
        model_config = load_yaml_config(model_config_path)
        vocab_bundle = load_vocab_bundle(data_config)
        lab_feature_size, vital_feature_size = _infer_numeric_feature_sizes(data_config_path)

        hidden_dim = int(model_config.get("model", {}).get("hidden_dim", 128))
        model_dropout = float(model_config.get("model", {}).get("dropout", 0.1))
        embedding_cfg = dict(model_config.get("embedding", {}))
        history_cfg = dict(model_config.get("history_selector", {}))
        fusion_cfg = dict(model_config.get("fusion", {}))

        code_embedding_dim = int(embedding_cfg.get("diag_dim", hidden_dim))
        proc_dim = int(embedding_cfg.get("proc_dim", code_embedding_dim))
        if proc_dim != code_embedding_dim:
            raise ValueError(
                "PatientStateEncoder requires matching diagnosis/procedure embedding widths in the new core pipeline."
            )

        numeric_projection_dim = int(embedding_cfg.get("lab_dim", 64))
        vital_dim = int(embedding_cfg.get("vital_dim", numeric_projection_dim))
        if vital_dim != numeric_projection_dim:
            raise ValueError(
                "PatientStateEncoder requires matching lab/vital projection widths in the new core pipeline."
            )

        ddi_matrix, lambda_ddi = _load_optional_ddi_matrix(train_config_path)
        encoder = PatientStateEncoder(
            diagnosis_vocab_size=len(vocab_bundle["diagnosis"]["idx_to_token"]),
            procedure_vocab_size=len(vocab_bundle["procedure"]["idx_to_token"]),
            drug_vocab_size=len(vocab_bundle["drug"]["idx_to_token"]),
            num_lab_features=lab_feature_size,
            num_vital_features=vital_feature_size,
            code_embedding_dim=code_embedding_dim,
            medication_embedding_dim=int(embedding_cfg.get("drug_dim", hidden_dim)),
            numeric_projection_dim=numeric_projection_dim,
            time_embedding_dim=int(embedding_cfg.get("time_dim", 32)),
            visit_hidden_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=model_dropout,
        )
        history_selector = SelfHistorySelector(
            hidden_dim=hidden_dim,
            dropout=float(history_cfg.get("dropout", model_dropout)),
            self_top_k=history_cfg.get("self_top_k"),
        )
        fusion_module = FusionModule(
            hidden_dim=hidden_dim,
            dropout=float(fusion_cfg.get("dropout", model_dropout)),
            strategy=str(fusion_cfg.get("strategy", "gated")),
        )
        decoder = MedicationDecoder(
            hidden_dim=hidden_dim,
            drug_vocab_size=len(vocab_bundle["drug"]["idx_to_token"]),
            dropout=model_dropout,
        )
        return cls(
            encoder,
            history_selector,
            fusion_module,
            medication_decoder=decoder,
            ddi_matrix=ddi_matrix,
            lambda_ddi=lambda_ddi,
        )

    def _resolve_ddi_matrix(self, batch: Mapping[str, Any]) -> torch.Tensor | None:
        ddi_adj = batch.get("ddi_adj")
        if ddi_adj is None:
            return self.ddi_matrix
        return torch.as_tensor(ddi_adj, dtype=torch.float32)

    def forward(self, batch: Mapping[str, Any], **_: Any) -> dict[str, Any]:
        enc_out = self.encoder(dict(batch))
        state_sequence = enc_out["state_sequence"]                   # [B, T, H]
        visit_mask = enc_out["visit_mask"]                           # [B, T]
        current_state = extract_last_valid_state(state_sequence, visit_mask)  # [B, H]

        sel_out = self.self_history_selector(
            current_state=current_state,
            state_sequence=state_sequence,
            visit_mask=visit_mask,
        )
        fusion_out = self.fusion_module(
            current_state=current_state,
            self_history_summary=sel_out["self_history_summary"],
        )

        if self.medication_decoder is None:
            raise RuntimeError("FullMedicationModel requires a MedicationDecoder for forward inference.")
        dec_out = self.medication_decoder(fusion_out["context_vector"])

        target_drugs = batch.get("target_drugs")
        target_current: torch.Tensor | None = None
        prediction_loss: torch.Tensor | None = None
        ddi_loss: torch.Tensor | None = None
        weighted_ddi_loss: torch.Tensor | None = None
        total_loss: torch.Tensor | None = None

        if target_drugs is not None:
            target_tensor = torch.as_tensor(
                target_drugs,
                device=dec_out["drug_logits"].device,
                dtype=dec_out["drug_logits"].dtype,
            )
            target_current = extract_last_valid_targets(target_tensor, visit_mask)
            loss_outputs = compute_medication_losses(
                drug_logits=dec_out["drug_logits"],
                drug_probs=dec_out["drug_probs"],
                target_drugs=target_tensor,
                visit_mask=visit_mask,
                ddi_matrix=self._resolve_ddi_matrix(batch),
                lambda_ddi=self.lambda_ddi,
                reduction="mean",
            )
            prediction_loss = loss_outputs["prediction_loss"]
            ddi_loss = loss_outputs["ddi_loss"]
            weighted_ddi_loss = loss_outputs["weighted_ddi_loss"]
            total_loss = loss_outputs["total_loss"]
            target_current = loss_outputs["target_current"]

        return {
            "visit_repr": enc_out["visit_repr"],
            "state_sequence": state_sequence,
            "pooled_state": enc_out["pooled_state"],
            "visit_mask": visit_mask,
            "current_state": current_state,
            "self_history_summary": sel_out["self_history_summary"],
            "self_attention_weights": sel_out["self_attention_weights"],
            "selected_visit_indices": sel_out["selected_visit_indices"],
            "context_vector": fusion_out["context_vector"],
            "drug_logits": dec_out["drug_logits"],
            "drug_probs": dec_out["drug_probs"],
            "target_current": target_current,
            "final_target_drugs": target_current,
            "prediction_loss": prediction_loss,
            "ddi_loss": ddi_loss,
            "weighted_ddi_loss": weighted_ddi_loss,
            "total_loss": total_loss,
        }

__all__ = ["FullMedicationModel", "extract_last_valid_state"]
