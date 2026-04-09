from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_core_metrics
from src.training.losses import extract_last_valid_targets

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.utils.io import ensure_dir


_LOSS_KEYS = ("total_loss", "prediction_loss", "ddi_loss")
_TIME_KEYS = ("data_time", "step_time")


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor for logging, got shape {tuple(value.shape)}")
        return float(value.detach().cpu().item())
    return float(value)


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _log_line(message: str) -> None:
    writer = getattr(tqdm, "write", None)
    if callable(writer):
        writer(message)
        return
    print(message)


class Trainer:
    """Trainer for the self-history-only core pipeline."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        checkpoint_dir: str | Path,
        log_dir: str | Path,
        scheduler: Any | None = None,
        monitor_metric: str = "val_total_loss",
        monitor_mode: str = "min",
        decoder_top_k: int | None = None,
        loss_fn: nn.Module | None = None,
        validation_threshold: float = 0.5,
    ) -> None:
        if monitor_mode not in {"min", "max"}:
            raise ValueError(f"monitor_mode must be 'min' or 'max', got {monitor_mode!r}")
        if not 0.0 <= float(validation_threshold) <= 1.0:
            raise ValueError(f"validation_threshold must be in [0, 1], got {validation_threshold!r}")

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.monitor_metric = str(monitor_metric)
        self.monitor_mode = str(monitor_mode)
        self.decoder_top_k = decoder_top_k
        self.loss_fn = loss_fn
        self.validation_threshold = float(validation_threshold)

        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.log_dir = ensure_dir(log_dir)
        self.best_checkpoint_path = self.checkpoint_dir / "train_core_best.pt"
        self.metrics_log_path = self.log_dir / "train_core_metrics.jsonl"
        self.best_metric = float("inf") if monitor_mode == "min" else float("-inf")

    def _resolve_validation_ddi_matrix(self, batch: Mapping[str, Any]) -> torch.Tensor | None:
        ddi_matrix = batch.get("ddi_adj")
        if ddi_matrix is None:
            ddi_matrix = getattr(self.model, "ddi_matrix", None)
        if ddi_matrix is None:
            return None

        resolved = torch.as_tensor(ddi_matrix, dtype=torch.float32).detach().cpu()
        if resolved.ndim != 2:
            raise ValueError(f"Validation ddi_matrix must have shape (D, D), got {tuple(resolved.shape)}")
        return resolved

    def _resolve_validation_targets(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
    ) -> torch.Tensor | None:
        target_current = outputs.get("final_target_drugs")
        if target_current is None:
            target_current = outputs.get("target_current")
        if target_current is None:
            target_current = batch.get("target_drugs")
            if isinstance(target_current, torch.Tensor) and target_current.ndim == 3:
                visit_mask = batch.get("visit_mask")
                if not isinstance(visit_mask, torch.Tensor):
                    raise RuntimeError("visit_mask is required to resolve validation targets from [B, T, D].")
                target_current = extract_last_valid_targets(target_current, visit_mask)
        return None if target_current is None else torch.as_tensor(target_current)

    def _run_one_epoch(
        self,
        dataloader: DataLoader,
        *,
        training: bool,
    ) -> dict[str, float]:
        phase = "train" if training else "val"
        totals = {key: 0.0 for key in _LOSS_KEYS}
        timing_totals = {key: 0.0 for key in _TIME_KEYS}
        total_examples = 0
        total_batches = 0
        collected_probs: list[torch.Tensor] = []
        collected_targets: list[torch.Tensor] = []
        validation_ddi_matrix: torch.Tensor | None = None

        self.model.train(mode=training)
        grad_context = torch.enable_grad if training else torch.no_grad
        progress = tqdm(
            range(len(dataloader)),
            desc=f"{phase} batches",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            mininterval=2.0,
        )

        dataloader_iter = iter(dataloader)
        for step_index in progress:
            data_start = time.perf_counter()
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break
            data_time = time.perf_counter() - data_start

            step_start = time.perf_counter()
            batch_on_device = _move_batch_to_device(batch, self.device)
            batch_size = int(batch_on_device["visit_mask"].shape[0])
            if batch_size <= 0:
                continue

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with grad_context():
                outputs = self.model(batch_on_device)
                total_loss = outputs.get("total_loss")
                prediction_loss = outputs.get("prediction_loss")
                ddi_loss = outputs.get("ddi_loss")
                if total_loss is None or prediction_loss is None or ddi_loss is None:
                    raise RuntimeError(
                        "Model forward must return `total_loss`, `prediction_loss`, and `ddi_loss` "
                        "for the new training pipeline."
                    )
                if not training:
                    drug_probs = outputs.get("drug_probs")
                    target_current = self._resolve_validation_targets(outputs, batch_on_device)
                    if drug_probs is None or target_current is None:
                        raise RuntimeError(
                            "Model forward must return `drug_probs` and current-visit targets for validation metrics."
                        )
                    collected_probs.append(drug_probs.detach().cpu())
                    collected_targets.append(target_current.detach().cpu())
                    batch_ddi_matrix = self._resolve_validation_ddi_matrix(batch_on_device)
                    if batch_ddi_matrix is None:
                        raise RuntimeError("Validation metrics require an available ddi_matrix.")
                    if validation_ddi_matrix is None:
                        validation_ddi_matrix = batch_ddi_matrix
                    elif not torch.equal(validation_ddi_matrix, batch_ddi_matrix):
                        raise ValueError("Validation batches produced inconsistent ddi_matrix values.")

                if training:
                    total_loss.backward()
                    self.optimizer.step()
            _synchronize_device(self.device)
            step_time = time.perf_counter() - step_start

            total_batches += 1
            total_examples += batch_size
            totals["total_loss"] += _to_float(total_loss) * batch_size
            totals["prediction_loss"] += _to_float(prediction_loss) * batch_size
            totals["ddi_loss"] += _to_float(ddi_loss) * batch_size
            timing_totals["data_time"] += data_time
            timing_totals["step_time"] += step_time
            batch_count = max(total_batches, 1)
            if hasattr(progress, "set_postfix") and (total_batches == 1 or total_batches % 20 == 0):
                progress.set_postfix(
                    total_loss=f"{totals['total_loss'] / float(total_examples):.4f}",
                    pred_loss=f"{totals['prediction_loss'] / float(total_examples):.4f}",
                    ddi_loss=f"{totals['ddi_loss'] / float(total_examples):.4f}",
                    data_time=f"{timing_totals['data_time'] / float(batch_count):.3f}s",
                    step_time=f"{timing_totals['step_time'] / float(batch_count):.3f}s",
                )

        if total_examples <= 0:
            raise ValueError(f"{phase} dataloader produced zero valid examples")

        metrics = {f"{phase}_{key}": totals[key] / float(total_examples) for key in _LOSS_KEYS}
        average_batches = float(max(total_batches, 1))
        metrics.update({f"{phase}_{key}": timing_totals[key] / average_batches for key in _TIME_KEYS})
        if not training:
            if not collected_probs or not collected_targets or validation_ddi_matrix is None:
                raise ValueError("Validation epoch did not produce metric inputs")
            validation_metrics = compute_core_metrics(
                y_true=torch.cat(collected_targets, dim=0),
                y_score=torch.cat(collected_probs, dim=0),
                threshold=self.validation_threshold,
                ddi_matrix=validation_ddi_matrix,
            )
            metrics.update(
                {
                    "val_jaccard": float(validation_metrics["jaccard"]),
                    "val_f1": float(validation_metrics["f1"]),
                    "val_prauc": float(validation_metrics["prauc"]),
                    "val_ddi_rate": float(validation_metrics["ddi_rate"]),
                }
            )
        return metrics

    def train_one_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        return self._run_one_epoch(dataloader, training=True)

    def validate_one_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        return self._run_one_epoch(dataloader, training=False)

    def save_best_checkpoint(
        self,
        *,
        epoch: int,
        epoch_metrics: Mapping[str, float],
        extra_state: Mapping[str, Any] | None = None,
    ) -> Path | None:
        if self.monitor_metric not in epoch_metrics:
            raise KeyError(f"Missing monitor metric `{self.monitor_metric}` in epoch metrics")

        current_metric = float(epoch_metrics[self.monitor_metric])
        is_better = (
            current_metric < self.best_metric
            if self.monitor_mode == "min"
            else current_metric > self.best_metric
        )
        if not is_better:
            return None

        self.best_metric = current_metric
        checkpoint_payload: dict[str, Any] = {
            "epoch": int(epoch),
            "best_metric": current_metric,
            "monitor_metric": self.monitor_metric,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint_payload["scheduler_state_dict"] = self.scheduler.state_dict()
        if extra_state:
            checkpoint_payload.update(dict(extra_state))

        torch.save(checkpoint_payload, self.best_checkpoint_path)
        return self.best_checkpoint_path

    def log_metrics(self, *, epoch: int, metrics: Mapping[str, Any]) -> None:
        bottleneck = (
            "data_loader"
            if float(metrics["train_data_time"]) > float(metrics["train_step_time"])
            else "model_step"
        )
        summary = (
            f"Epoch {epoch}: "
            f"train_total_loss={float(metrics['train_total_loss']):.6f} "
            f"train_prediction_loss={float(metrics['train_prediction_loss']):.6f} "
            f"train_ddi_loss={float(metrics['train_ddi_loss']):.6f} "
            f"train_data_time={float(metrics['train_data_time']):.3f}s "
            f"train_step_time={float(metrics['train_step_time']):.3f}s "
            f"val_total_loss={float(metrics['val_total_loss']):.6f} "
            f"val_prediction_loss={float(metrics['val_prediction_loss']):.6f} "
            f"val_ddi_loss={float(metrics['val_ddi_loss']):.6f} "
            f"val_jaccard={float(metrics['val_jaccard']):.6f} "
            f"val_f1={float(metrics['val_f1']):.6f} "
            f"val_prauc={float(metrics['val_prauc']):.6f} "
            f"val_ddi_rate={float(metrics['val_ddi_rate']):.6f} "
            f"val_data_time={float(metrics['val_data_time']):.3f}s "
            f"val_step_time={float(metrics['val_step_time']):.3f}s "
            f"bottleneck={bottleneck}"
        )
        _log_line(summary)
        log_payload = {"epoch": int(epoch), **{key: _to_float(value) for key, value in metrics.items()}}
        with self.metrics_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_payload, ensure_ascii=True, sort_keys=True))
            handle.write("\n")

    def fit(
        self,
        *,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int,
        extra_checkpoint_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if int(epochs) <= 0:
            raise ValueError(f"epochs must be positive, got {epochs!r}")

        history: list[dict[str, float]] = []
        best_checkpoint_path: Path | None = None

        epoch_progress = tqdm(
            range(1, int(epochs) + 1),
            desc="Training epochs",
            unit="epoch",
            dynamic_ncols=True,
            mininterval=1.0,
        )
        for epoch in epoch_progress:
            train_metrics = self.train_one_epoch(train_dataloader)
            val_metrics = self.validate_one_epoch(val_dataloader)
            epoch_metrics = {**train_metrics, **val_metrics}

            if self.scheduler is not None:
                self.scheduler.step()

            maybe_best = self.save_best_checkpoint(
                epoch=epoch,
                epoch_metrics=epoch_metrics,
                extra_state=extra_checkpoint_state,
            )
            if maybe_best is not None:
                best_checkpoint_path = maybe_best

            if hasattr(epoch_progress, "set_postfix"):
                epoch_progress.set_postfix(
                    train_total=f"{float(epoch_metrics['train_total_loss']):.4f}",
                    val_total=f"{float(epoch_metrics['val_total_loss']):.4f}",
                    val_jac=f"{float(epoch_metrics['val_jaccard']):.4f}",
                    val_ddi=f"{float(epoch_metrics['val_ddi_rate']):.4f}",
                    data=f"{float(epoch_metrics['train_data_time']):.3f}s",
                    step=f"{float(epoch_metrics['train_step_time']):.3f}s",
                )
            self.log_metrics(epoch=epoch, metrics=epoch_metrics)
            history.append({"epoch": float(epoch), **epoch_metrics})

        return {
            "history": history,
            "best_metric": self.best_metric,
            "best_checkpoint_path": None if best_checkpoint_path is None else str(best_checkpoint_path),
            "monitor_metric": self.monitor_metric,
        }


__all__ = ["Trainer", "_LOSS_KEYS", "_move_batch_to_device", "_to_float"]
