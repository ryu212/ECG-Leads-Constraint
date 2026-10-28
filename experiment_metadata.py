"""Shared reproducibility and run-metadata helpers for training entrypoints."""

from __future__ import annotations

import csv
import hashlib
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


RUN_METADATA_COLUMNS = (
    "run_id",
    "status",
    "started_at_utc",
    "finished_at_utc",
    "error",
    "version",
    "seed",
    "trainpath",
    "valpath",
    "modelpath",
    "best_checkpoint",
    "last_checkpoint",
    "resume_from",
    "start_epoch",
    "completed_epoch",
    "best_val_loss",
    "epochs",
    "batch_size",
    "lr",
    "ecglen",
    "kernel_size",
    "pool_size",
    "anylead",
    "patience",
    "figure_plot",
    "stage1_epochs",
    "alpha",
    "lambda",
    "smoke_test",
    "fixed_beta",
    "multirate",
    "multirate_rates_hz",
    "last_train_loss",
    "last_train_pcc",
    "last_val_loss",
    "last_val_pcc",
    "optimizer",
    "source_version",
    "source_dirty",
    "device",
)

HYPERPARAMETER_FIELDS = (
    "epochs",
    "batch_size",
    "lr",
    "ecglen",
    "kernel_size",
    "pool_size",
    "anylead",
    "patience",
    "figure_plot",
    "stage1_epochs",
    "alpha",
    "lambda",
    "fixed_beta",
    "multirate",
    "multirate_rates_hz",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed Python, NumPy, Torch and all visible CUDA devices."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _git_source_version(repo_root: Path) -> tuple[str, bool] | None:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return f"git:{revision}", bool(status_result.stdout.strip())


def _manifest_source_version(repo_root: Path) -> tuple[str, bool] | None:
    manifest = repo_root / "artifacts/manifests/source_manifest.csv"
    if not manifest.is_file():
        return None
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return f"manifest-sha256:{digest}", False


def resolve_source_version(
    repo_root: Path, explicit: str | None = None
) -> tuple[str, bool | None]:
    """Resolve an explicit, Git, or source-manifest version identifier."""
    if explicit:
        return explicit, None
    git_version = _git_source_version(repo_root)
    if git_version is not None:
        return git_version
    manifest_version = _manifest_source_version(repo_root)
    if manifest_version is not None:
        return manifest_version
    raise RuntimeError(
        "Cannot determine source version. Provide --source-version or restore "
        "artifacts/manifests/source_manifest.csv."
    )


def build_run_metadata(
    *,
    args,
    version: str,
    source_version: str,
    source_dirty: bool | None,
    optimizer: str,
    device: str,
) -> dict[str, Any]:
    started_at = _utc_now()
    seed = int(args.seed)
    return {
        "run_id": f"{version}-seed{seed}-{started_at}",
        "status": "running",
        "started_at_utc": started_at,
        "finished_at_utc": "",
        "error": "",
        "version": version,
        "seed": seed,
        "trainpath": Path(args.trainpath).as_posix(),
        "valpath": Path(args.valpath).as_posix(),
        "modelpath": Path(args.modelpath).as_posix(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "ecglen": args.ecglen,
        "kernel_size": args.kernel_size,
        "pool_size": args.pool_size,
        "anylead": args.anylead,
        "patience": args.patience,
        "figure_plot": getattr(args, "figure_plot", 0),
        "stage1_epochs": getattr(args, "stage1_epochs", ""),
        "alpha": getattr(args, "alpha", ""),
        "lambda": getattr(args, "lambda_", ""),
        "smoke_test": getattr(args, "smoke_test", False),
        "fixed_beta": getattr(args, "fixed_beta", ""),
        "multirate": getattr(args, "multirate", False),
        "multirate_rates_hz": getattr(args, "multirate_rates_hz", ""),
        "optimizer": optimizer,
        "source_version": source_version,
        "source_dirty": "" if source_dirty is None else source_dirty,
        "device": device,
    }


def metadata_path_for(modelpath: Path) -> Path:
    return Path(modelpath).parent / "run_metadata.csv"


def write_run_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_METADATA_COLUMNS)
        writer.writeheader()
        writer.writerow({column: metadata.get(column, "") for column in RUN_METADATA_COLUMNS})


def checkpoint_metadata_fields(
    metadata: dict[str, Any] | None,
    optimizer: torch.optim.Optimizer,
    epoch: int | None,
) -> dict[str, Any]:
    metadata = metadata or {}
    hyperparameters = {
        field: metadata.get(field, "") for field in HYPERPARAMETER_FIELDS
    }
    return {
        "epoch": epoch,
        "seed": metadata.get("seed"),
        "version": metadata.get("version", ""),
        "hyperparameters": hyperparameters,
        "optimizer": {
            "name": optimizer.__class__.__name__,
            "defaults": dict(optimizer.defaults),
        },
        "source_version": metadata.get("source_version", ""),
        "source_dirty": metadata.get("source_dirty", ""),
        "run_id": metadata.get("run_id", ""),
    }


def train_with_metadata(
    trainer,
    train_loader,
    val_loader,
    metadata: dict[str, Any],
    metadata_path: Path,
) -> None:
    """Run the existing trainer while recording lifecycle metadata."""
    trainer.run_metadata = metadata
    write_run_metadata(metadata, metadata_path)
    try:
        trainer.train(train_loader, val_loader)
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["finished_at_utc"] = _utc_now()
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        write_run_metadata(metadata, metadata_path)
        raise
    metadata["status"] = "success"
    metadata["finished_at_utc"] = _utc_now()
    write_run_metadata(metadata, metadata_path)
