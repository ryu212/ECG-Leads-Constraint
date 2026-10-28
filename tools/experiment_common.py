"""Shared model loading and ECG metric contracts for experiment tools."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
DERIVED_INDICES = (2, 3, 4, 5)
DERIVED_NAMES = tuple(LEAD_NAMES[index] for index in DERIVED_INDICES)
TUPLE_OUTPUT_VERSIONS = frozenset({"v8", "v12", "v13", "v14", "v15"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_payload(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    value = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint must be a dict: {path}")
    return value


def checkpoint_config(path: str | Path) -> dict[str, Any]:
    payload = checkpoint_payload(path)
    config = payload.get("config", {})
    return config if isinstance(config, dict) else {}


def prediction_and_base(output: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    if torch.is_tensor(output):
        return output, None
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        base = output[1] if len(output) > 1 and torch.is_tensor(output[1]) else None
        return output[0], base
    raise TypeError("Model output must be a Tensor or a tuple/list beginning with Tensor")


def load_model(
    version: str,
    modelpath: str | Path,
    *,
    device: str | torch.device | None = None,
    ecglen: int = 1024,
    kernel_size: int = 5,
    pool_size: int = 2,
    alpha: float | None = None,
    lambda_: float | None = None,
):
    """Instantiate the matching Trainer and strict-load one checkpoint."""
    modelpath = Path(modelpath)
    if not modelpath.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {modelpath}")
    module = importlib.import_module(f"{version}.utils_MCMA")
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = checkpoint_config(modelpath)
    kwargs: dict[str, Any] = {
        "modelpath": modelpath,
        "ecglen": int(config.get("ecglen", ecglen)),
        "kernel_size": int(config.get("kernel_size", kernel_size)),
        "pool_size": int(config.get("pool_size", pool_size)),
        "device": selected_device,
    }
    if version not in {"v0", "v1"}:
        kwargs["alpha"] = float(alpha if alpha is not None else config.get("alpha", 0.5))
    if version in TUPLE_OUTPUT_VERSIONS:
        kwargs["stage1_epochs"] = int(config.get("stage1_epochs", 50))
    if version == "v15":
        resolved_lambda = lambda_ if lambda_ is not None else config.get("lambda")
        if resolved_lambda in (None, ""):
            raise ValueError("A v15 checkpoint without config requires --lambda")
        kwargs["lambda_"] = float(resolved_lambda)
    trainer = module.Trainer(**kwargs)
    trainer.load(modelpath)
    return trainer.model.to(selected_device).eval(), module, selected_device, config


def pearson_per_record(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_centered = pred - pred.mean(dim=-1, keepdim=True)
    true_centered = true - true.mean(dim=-1, keepdim=True)
    numerator = (pred_centered * true_centered).mean(dim=-1)
    denominator = pred.std(dim=-1, unbiased=False) * true.std(dim=-1, unbiased=False)
    return torch.nan_to_num(numerator / (denominator + eps), nan=0.0, posinf=0.0, neginf=0.0)


def derive_limb(y12: torch.Tensor) -> torch.Tensor:
    lead_i, lead_ii = y12[:, 0], y12[:, 1]
    return torch.stack(
        (
            lead_ii - lead_i,
            -0.5 * (lead_i + lead_ii),
            lead_i - 0.5 * lead_ii,
            lead_ii - 0.5 * lead_i,
        ),
        dim=1,
    )


def eg_violation_per_record(y12: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    direct = y12[:, 2:6]
    expected = derive_limb(y12)
    numerator = (direct - expected).pow(2).mean(dim=-1).sqrt()
    denominator = direct.pow(2).mean(dim=-1).sqrt() + eps
    return numerator / denominator


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer CSV fields for empty output: {path}")
        fieldnames = rows[0].keys()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_metadata(
    path: str | Path,
    *,
    version: str,
    modelpath: str | Path,
    dataset: str | Path,
    tag: str,
    extra: dict[str, Any] | None = None,
) -> None:
    row: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "tag": tag,
        "version": version,
        "modelpath": Path(modelpath).as_posix(),
        "model_sha256": sha256(modelpath),
        "dataset": Path(dataset).as_posix(),
        "dataset_sha256": sha256(dataset),
    }
    if extra:
        row.update({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in extra.items()})
    write_csv(path, [row])

