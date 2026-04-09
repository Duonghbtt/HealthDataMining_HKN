from __future__ import annotations

import copy
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import MIMICTrajectoryDataset, collate_batch
from src.data.tensorized_dataset import (
    TensorizedTrajectoryDataset,
    tensorized_collate_batch,
    tensorized_manifest_path_from_config,
)
from src.models.ddi_regularization import load_ddi_matrix
from src.models.full_model import FullMedicationModel
from src.models.fusion import FusionModule
from src.models.history_selector import SelfHistorySelector
from src.models.medication_decoder import MedicationDecoder
from src.models.patient_state_encoder import PatientStateEncoder
from src.utils.io import ensure_dir, load_yaml_config, read_json, resolve_path


class DirectParquetTrajectoryDataset(Dataset):
    """Fallback dataset for direct split manifest layout under `processed/<split>`."""

    def __init__(
        self,
        split: str,
        processed_root: str | Path,
        *,
        drug_vocab_size: int,
        max_open_shards: int = 8,
    ) -> None:
        self.split = split
        self.processed_root = Path(processed_root)
        self.drug_vocab_size = int(drug_vocab_size)
        self.max_open_shards = int(max_open_shards)
        self._storage_mode = "direct_parquet"
        self.shards: list[dict[str, Any]] = []
        self.cumulative_rows: list[int] = []
        self._shard_cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self.default_lab_feature_size = 0
        self.default_vital_feature_size = 0

        manifest_path = self.processed_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing processed manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        split_payload = manifest.get("splits", {}).get(split)
        if split_payload is None:
            raise FileNotFoundError(f"Split `{split}` is missing from manifest {manifest_path}")

        metadata_path = self.processed_root / "metadata.json"
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            self.default_lab_feature_size = int(metadata.get("lab_feature_size", 0))
            self.default_vital_feature_size = int(metadata.get("vital_feature_size", 0))

        total = 0
        for shard in split_payload.get("shards", []):
            shard_path = self.processed_root / shard["path"]
            rows = int(shard["rows"])
            self.shards.append({"path": shard_path, "rows": rows})
            total += rows
            self.cumulative_rows.append(total)

    @property
    def storage_mode(self) -> str:
        return self._storage_mode

    @property
    def num_shards(self) -> int:
        return len(self.shards)

    def __len__(self) -> int:
        return self.cumulative_rows[-1] if self.cumulative_rows else 0

    def _augment_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        resolved = dict(record)
        steps = list(resolved.get("steps", []))
        resolved["drug_vocab_size"] = int(resolved.get("drug_vocab_size", self.drug_vocab_size))
        resolved["num_steps"] = int(resolved.get("num_steps", len(steps)))
        resolved["lab_feature_size"] = int(
            resolved.get(
                "lab_feature_size",
                max((len(step.get("lab_values", [])) for step in steps), default=self.default_lab_feature_size),
            )
        )
        resolved["vital_feature_size"] = int(
            resolved.get(
                "vital_feature_size",
                max((len(step.get("vital_values", [])) for step in steps), default=self.default_vital_feature_size),
            )
        )
        return resolved

    def _touch_cached_shard(self, shard_index: int) -> list[dict[str, Any]] | None:
        cached_rows = self._shard_cache.pop(shard_index, None)
        if cached_rows is not None:
            self._shard_cache[shard_index] = cached_rows
        return cached_rows

    def _store_cached_shard(self, shard_index: int, rows: list[dict[str, Any]]) -> None:
        self._shard_cache[shard_index] = rows
        while len(self._shard_cache) > self.max_open_shards:
            self._shard_cache.popitem(last=False)

    def _load_shard(self, shard_index: int) -> list[dict[str, Any]]:
        cached_rows = self._touch_cached_shard(shard_index)
        if cached_rows is not None:
            return cached_rows

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required to read parquet trajectories. Install requirements.txt first."
            ) from exc

        shard = self.shards[shard_index]
        shard_path = Path(shard["path"])
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing trajectory shard: {shard_path}")

        rows = [self._augment_record(row) for row in pq.read_table(shard_path, use_threads=True).to_pylist()]
        if len(rows) != int(shard["rows"]):
            raise RuntimeError(
                f"Shard row count mismatch at {shard_path}: manifest={shard['rows']} actual={len(rows)}"
            )

        self._store_cached_shard(shard_index, rows)
        return rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative_rows, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_rows[shard_index - 1]
        local_index = index - shard_start
        return dict(self._load_shard(shard_index)[local_index])


def resolve_device(requested_device: str) -> torch.device:
    device = torch.device(str(requested_device))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is not available; falling back to CPU.")
        return torch.device("cpu")
    return device


def _runtime_cache_size(runtime_data_config_path: Path) -> int:
    runtime_data_config = load_yaml_config(runtime_data_config_path)
    spark_cfg = runtime_data_config.get("spark", {})
    if isinstance(spark_cfg, dict) and spark_cfg.get("max_open_shards_per_dataset") is not None:
        return int(spark_cfg["max_open_shards_per_dataset"])
    return 8


def _dataset_storage_mode(dataset: Dataset) -> str:
    return str(
        getattr(
            dataset,
            "storage_mode",
            getattr(dataset, "_storage_mode", "unknown"),
        )
    )


def _dataset_num_shards(dataset: Dataset) -> int:
    num_shards = getattr(dataset, "num_shards", None)
    if num_shards is not None:
        return int(num_shards)
    shards = getattr(dataset, "shards", None)
    if isinstance(shards, Sequence):
        return len(shards)
    return 0


def _print_dataset_details(split: str, dataset: Dataset) -> None:
    print(
        f"Dataset `{split}`: "
        f"class={type(dataset).__name__} "
        f"storage_mode={_dataset_storage_mode(dataset)} "
        f"size={len(dataset)} "
        f"num_shards={_dataset_num_shards(dataset)} "
        f"max_open_shards={getattr(dataset, 'max_open_shards', 'n/a')}"
    )


def select_collate_fn(dataset: Dataset):
    if isinstance(dataset, TensorizedTrajectoryDataset) or _dataset_storage_mode(dataset) == "tensorized_pt":
        return tensorized_collate_batch
    return collate_batch


def build_runtime_data_config_file(
    *,
    project_root: Path,
    data_config: Mapping[str, Any],
    processed_root: Path,
    vocab_root: Path,
    temp_dir: Path,
) -> Path:
    runtime_config = copy.deepcopy({key: value for key, value in data_config.items() if not str(key).startswith("_")})
    runtime_config.setdefault("paths", {})
    runtime_paths = runtime_config["paths"]
    runtime_paths["processed_root"] = str(processed_root.resolve())
    runtime_paths["interim_root"] = str(vocab_root.parent.resolve())
    runtime_paths["vocab_root"] = str(vocab_root.resolve())

    for path_key in (
        "raw_root",
        "cohort_root",
        "trajectory_interim_root",
        "ddi_root",
        "artifacts_root",
        "tensorized_root",
        "encoder_artifact_root",
        "ddi_source_path",
    ):
        path_value = runtime_paths.get(path_key)
        if path_value:
            runtime_paths[path_key] = str(resolve_path(project_root, path_value).resolve())

    spark_cfg = runtime_config.get("spark", {})
    if isinstance(spark_cfg, dict):
        spark_cfg = dict(spark_cfg)
        stage_cache_dir = spark_cfg.get("stage_cache_dir")
        if stage_cache_dir:
            spark_cfg["stage_cache_dir"] = str(resolve_path(project_root, stage_cache_dir).resolve())
        runtime_config["spark"] = spark_cfg

    runtime_config_path = temp_dir / "runtime_data.yaml"
    runtime_config_path.write_text(yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8")
    return runtime_config_path


def build_dataset(
    *,
    split: str,
    runtime_data_config_path: Path,
    processed_root: Path,
    drug_vocab_size: int,
    dataset_cache_size: int | None = None,
) -> Dataset:
    resolved_dataset_cache_size = int(
        dataset_cache_size if dataset_cache_size is not None else _runtime_cache_size(runtime_data_config_path)
    )
    runtime_config = load_yaml_config(runtime_data_config_path)
    tensorized_manifest_path = tensorized_manifest_path_from_config(runtime_config)
    if tensorized_manifest_path.exists():
        try:
            dataset = TensorizedTrajectoryDataset(
                split,
                runtime_data_config_path,
                max_open_shards=resolved_dataset_cache_size,
            )
            print(
                f"Using TensorizedTrajectoryDataset for split `{split}` "
                f"(size={len(dataset)}, manifest={tensorized_manifest_path})"
            )
            return dataset
        except FileNotFoundError as exc:
            print(f"Tensorized dataset unavailable for split `{split}`: {exc}")

    try:
        dataset = MIMICTrajectoryDataset(split, runtime_data_config_path)
        print(
            f"Using MIMICTrajectoryDataset for split `{split}` "
            f"(size={len(dataset)}, config={runtime_data_config_path})"
        )
        return dataset
    except FileNotFoundError as exc:
        manifest_path = processed_root / "manifest.json"
        if not manifest_path.exists():
            raise
        print(f"Falling back to direct parquet dataset for split `{split}`: {exc}")
        dataset = DirectParquetTrajectoryDataset(
            split,
            processed_root,
            drug_vocab_size=drug_vocab_size,
            max_open_shards=resolved_dataset_cache_size,
        )
        print(
            f"Using DirectParquetTrajectoryDataset for split `{split}` "
            f"(size={len(dataset)}, processed_root={processed_root})"
        )
        return dataset


def build_dataloaders(
    *,
    runtime_data_config_path: Path,
    processed_root: Path,
    drug_vocab_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, Dataset]:
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}")
    if int(num_workers) < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers!r}")

    dataset_cache_size = _runtime_cache_size(runtime_data_config_path)
    train_dataset = build_dataset(
        split="train",
        runtime_data_config_path=runtime_data_config_path,
        processed_root=processed_root,
        drug_vocab_size=drug_vocab_size,
        dataset_cache_size=dataset_cache_size,
    )
    val_dataset = build_dataset(
        split="val",
        runtime_data_config_path=runtime_data_config_path,
        processed_root=processed_root,
        drug_vocab_size=drug_vocab_size,
        dataset_cache_size=dataset_cache_size,
    )
    if len(train_dataset) <= 0:
        raise ValueError("Training dataset is empty")
    if len(val_dataset) <= 0:
        raise ValueError("Validation dataset is empty")

    _print_dataset_details("train", train_dataset)
    _print_dataset_details("val", val_dataset)

    persistent_workers = int(num_workers) > 0
    prefetch_factor = 2 if persistent_workers else None
    print(
        "DataLoader settings: "
        f"batch_size={int(batch_size)} "
        f"num_workers={int(num_workers)} "
        f"pin_memory={bool(pin_memory)} "
        f"persistent_workers={persistent_workers} "
        f"prefetch_factor={prefetch_factor}"
    )

    loader_kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if persistent_workers:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=select_collate_fn(train_dataset),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=select_collate_fn(val_dataset),
        **loader_kwargs,
    )
    return train_loader, val_loader, train_dataset


def load_vocab_sizes(vocab_root: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for name in ("diagnosis", "procedure", "drug"):
        payload = read_json(vocab_root / f"{name}_vocab.json")
        sizes[name] = int(payload["size"])
    return sizes


def build_core_model(
    *,
    train_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    train_dataset: Dataset | None = None,
    runtime_data_config_path: Path | None = None,
    processed_root: Path | None = None,
    vocab_root: Path,
    ddi_matrix_path: Path,
) -> FullMedicationModel:
    vocab_sizes = load_vocab_sizes(vocab_root)
    resolved_train_dataset = train_dataset
    if resolved_train_dataset is None:
        if runtime_data_config_path is None or processed_root is None:
            raise ValueError("Either `train_dataset` or both `runtime_data_config_path` and `processed_root` must be provided.")
        resolved_train_dataset = build_dataset(
            split="train",
            runtime_data_config_path=runtime_data_config_path,
            processed_root=processed_root,
            drug_vocab_size=vocab_sizes["drug"],
        )

    num_lab_features = int(getattr(resolved_train_dataset, "default_lab_feature_size", 0))
    num_vital_features = int(getattr(resolved_train_dataset, "default_vital_feature_size", 0))
    if num_lab_features <= 0 or num_vital_features <= 0:
        sample_record = resolved_train_dataset[0]
        num_lab_features = num_lab_features or int(sample_record.get("lab_feature_size", 0))
        num_vital_features = num_vital_features or int(sample_record.get("vital_feature_size", 0))
        if (num_lab_features <= 0 or num_vital_features <= 0) and "lab_values" in sample_record and "vital_values" in sample_record:
            num_lab_features = num_lab_features or int(sample_record["lab_values"].shape[-1])
            num_vital_features = num_vital_features or int(sample_record["vital_values"].shape[-1])
        if num_lab_features <= 0 or num_vital_features <= 0:
            sample_batch = collate_batch([sample_record])
            num_lab_features = num_lab_features or int(sample_batch["lab_values"].shape[-1])
            num_vital_features = num_vital_features or int(sample_batch["vital_values"].shape[-1])

    model_cfg = dict(model_config.get("model", {}))
    embedding_cfg = dict(model_config.get("embedding", {}))
    history_cfg = dict(model_config.get("history_selector", {}))
    fusion_cfg = dict(model_config.get("fusion", {}))

    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    model_dropout = float(model_cfg.get("dropout", 0.1))
    code_embedding_dim = int(embedding_cfg.get("diag_dim", hidden_dim))
    proc_dim = int(embedding_cfg.get("proc_dim", code_embedding_dim))
    if proc_dim != code_embedding_dim:
        raise ValueError(
            "PatientStateEncoder currently expects a shared code embedding dim; "
            f"got diag_dim={code_embedding_dim}, proc_dim={proc_dim}"
        )
    numeric_projection_dim = int(embedding_cfg.get("lab_dim", 64))
    vital_dim = int(embedding_cfg.get("vital_dim", numeric_projection_dim))
    if vital_dim != numeric_projection_dim:
        raise ValueError(
            "PatientStateEncoder currently expects a shared numeric projection dim; "
            f"got lab_dim={numeric_projection_dim}, vital_dim={vital_dim}"
        )

    encoder = PatientStateEncoder(
        diagnosis_vocab_size=vocab_sizes["diagnosis"],
        procedure_vocab_size=vocab_sizes["procedure"],
        drug_vocab_size=vocab_sizes["drug"],
        num_lab_features=num_lab_features,
        num_vital_features=num_vital_features,
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
        dropout=float(history_cfg.get("dropout", 0.1)),
        self_top_k=history_cfg.get("self_top_k", 3),
    )
    fusion_module = FusionModule(
        hidden_dim=hidden_dim,
        dropout=float(fusion_cfg.get("dropout", model_dropout)),
        strategy=str(fusion_cfg.get("strategy", "gated")),
    )
    decoder = MedicationDecoder(
        hidden_dim=hidden_dim,
        drug_vocab_size=vocab_sizes["drug"],
        dropout=model_dropout,
    )
    ddi_matrix = load_ddi_matrix(ddi_matrix_path, device="cpu")
    lambda_ddi = float(train_config.get("loss", {}).get("lambda_ddi", 0.0))

    return FullMedicationModel(
        encoder,
        history_selector,
        fusion_module,
        medication_decoder=decoder,
        ddi_matrix=ddi_matrix,
        lambda_ddi=lambda_ddi,
    )


__all__ = [
    "DirectParquetTrajectoryDataset",
    "build_core_model",
    "build_dataloaders",
    "build_dataset",
    "build_runtime_data_config_file",
    "resolve_device",
    "select_collate_fn",
]
