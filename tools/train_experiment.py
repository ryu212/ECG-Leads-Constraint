"""Common, reproducible training entrypoint for MCMA experiment versions."""

from __future__ import annotations

import argparse
import importlib
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from experiment_metadata import (
    build_run_metadata,
    checkpoint_metadata_fields,
    metadata_path_for,
    resolve_source_version,
    seed_everything,
    write_run_metadata,
)
from project_paths import artifact_path, relative_path, require_repo_root


SINGLE_OUTPUT_VERSIONS = frozenset({"v0", "v1", "v3", "v4", "v7"})
CURRICULUM_VERSIONS = frozenset({"v8", "v12", "v13", "v14", "v15"})
AVAILABLE_VERSIONS = SINGLE_OUTPUT_VERSIONS | CURRICULUM_VERSIONS
PLANNED_VERSIONS = frozenset()
MULTIRATE_HZ = (250, 300, 400, 500)
SOURCE_HZ = 500


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def prediction_tensor(output: Any) -> torch.Tensor:
    """Return the prediction for models whose forward returns a tensor or tuple."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError("Model output must be a Tensor or a tuple/list starting with one")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainpath", required=True, type=relative_path)
    parser.add_argument("--valpath", required=True, type=relative_path)
    parser.add_argument("--modelpath", required=True, type=artifact_path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--lr", required=True, type=float)

    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=None)
    parser.add_argument("--stage1_epochs", type=int, default=None)
    parser.add_argument("--resume", type=relative_path, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--fixed_beta", type=float, default=None)
    multirate_group = parser.add_mutually_exclusive_group()
    multirate_group.add_argument(
        "--multirate",
        dest="multirate",
        action="store_true",
        help="Enable train-only 500 -> {250,300,400,500} -> 500 Hz augmentation.",
    )
    multirate_group.add_argument(
        "--no-multirate",
        dest="multirate",
        action="store_false",
        help="Disable multi-rate augmentation (the default).",
    )
    parser.set_defaults(multirate=False)

    parser.add_argument("--ecglen", type=int, default=1024)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--anylead", type=int, choices=(0, 1), default=1)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--source-version", default=None)
    return parser.parse_args(argv)


def _validate_positive_args(args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_size", "ecglen", "kernel_size", "pool_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.patience < 0 or args.num_workers < 0:
        raise ValueError("--patience and --num_workers must be non-negative")


def validate_and_resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate version-specific options and resolve protocol defaults."""
    _validate_positive_args(args)
    if args.modelpath.name != "best.pt":
        raise ValueError("--modelpath must name the best checkpoint file: best.pt")

    if args.version in PLANNED_VERSIONS:
        raise RuntimeError(
            f"PENDING_SOURCE: {args.version} has no runnable package in this repository"
        )
    if args.version not in AVAILABLE_VERSIONS:
        raise ValueError(
            f"Unsupported --version {args.version!r}; available: "
            f"{', '.join(sorted(AVAILABLE_VERSIONS))}"
        )

    if args.version in {"v0", "v1"}:
        if args.alpha is not None:
            raise ValueError(f"--alpha is not applicable to {args.version}")
    elif args.alpha is None:
        args.alpha = 0.5

    if args.version in CURRICULUM_VERSIONS:
        if args.stage1_epochs is None:
            args.stage1_epochs = 50
        if not 0 <= args.stage1_epochs <= args.epochs:
            raise ValueError("--stage1_epochs must be between 0 and --epochs")
    elif args.stage1_epochs is not None:
        raise ValueError(
            f"--stage1_epochs is only valid for {', '.join(sorted(CURRICULUM_VERSIONS))}"
        )

    if args.version == "v15":
        if args.lambda_ is None:
            raise ValueError("v15 requires --lambda")
        if args.lambda_ < 0:
            raise ValueError("--lambda must be non-negative")
    elif args.lambda_ is not None:
        raise ValueError("--lambda is only supported by v15")

    if args.fixed_beta is not None:
        if args.version != "v13":
            raise ValueError("--fixed_beta is only supported by v13")
        if not math.isclose(args.fixed_beta, 0.1, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("The controlled fixed-gate experiment requires --fixed_beta 0.1")

    if args.smoke_test:
        args.epochs = 1
        if args.version in CURRICULUM_VERSIONS:
            args.stage1_epochs = min(args.stage1_epochs, args.epochs)
        args.modelpath = Path(
            f"artifacts/smoke/{args.version}/seed_{args.seed}/best.pt"
        )
    return args


def require_cuda_device() -> torch.device:
    """Return the CUDA device or fail instead of silently training on CPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for training, but PyTorch cannot see CUDA. "
            "Install/run a CUDA-enabled PyTorch environment before launching this job."
        )
    return torch.device("cuda")


def load_version_module(version: str):
    if version not in AVAILABLE_VERSIONS:
        if version in PLANNED_VERSIONS:
            raise RuntimeError(f"PENDING_SOURCE: {version} is not implemented")
        raise ValueError(f"Unsupported version: {version}")
    return importlib.import_module(f"{version}.utils_MCMA")


def apply_multirate_batch(
    batch: torch.Tensor,
    rate_hz: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Round-trip one (B,T,C) train batch through a seeded random sample rate."""
    if not torch.is_tensor(batch) or batch.ndim != 3:
        raise ValueError("Multi-rate input must be a Tensor with shape (B,T,C)")
    selected_hz = random.choice(MULTIRATE_HZ) if rate_hz is None else int(rate_hz)
    if selected_hz not in MULTIRATE_HZ:
        raise ValueError(f"Unsupported multi-rate value: {selected_hz}")
    if selected_hz == SOURCE_HZ:
        return batch, selected_hz

    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError("--multirate requires SciPy (available in Kaggle)") from exc

    original_device = batch.device
    original_dtype = batch.dtype
    original_length = batch.shape[1]
    values = batch.detach().cpu().numpy()
    downsampled = resample_poly(values, selected_hz, SOURCE_HZ, axis=1)
    restored = resample_poly(downsampled, SOURCE_HZ, selected_hz, axis=1)

    if restored.shape[1] > original_length:
        restored = restored[:, :original_length, :]
    elif restored.shape[1] < original_length:
        restored = np.pad(
            restored,
            ((0, 0), (0, original_length - restored.shape[1]), (0, 0)),
            mode="constant",
        )
    if not np.isfinite(restored).all():
        raise FloatingPointError("Multi-rate augmentation produced NaN or Inf")
    result = torch.from_numpy(np.asarray(restored)).to(
        device=original_device, dtype=original_dtype
    )
    return result, selected_hz


class SingleOutputAdapter:
    """Adapter for legacy trainers whose model returns one prediction tensor."""

    @staticmethod
    def prepare_epoch(trainer, epoch: int) -> bool:
        return False

    @staticmethod
    def train_step(trainer, batch: torch.Tensor, is_stage1: bool):
        return trainer.train_step(batch)

    @staticmethod
    def val_step(trainer, batch: torch.Tensor, is_stage1: bool):
        return trainer.test_step(batch)


class CurriculumAdapter:
    """Adapter for tuple-output versions with two-stage training."""

    @staticmethod
    def prepare_epoch(trainer, epoch: int) -> bool:
        is_stage1 = epoch < trainer.stage1_epochs
        for parameter in trainer.model.residual_branch.parameters():
            parameter.requires_grad = not is_stage1
        return is_stage1

    @staticmethod
    def train_step(trainer, batch: torch.Tensor, is_stage1: bool):
        return trainer.train_step(batch, is_stage1)

    @staticmethod
    def val_step(trainer, batch: torch.Tensor, is_stage1: bool):
        return trainer.test_step(batch, is_stage1)


def adapter_for(version: str):
    if version in SINGLE_OUTPUT_VERSIONS:
        return SingleOutputAdapter
    if version in CURRICULUM_VERSIONS:
        return CurriculumAdapter
    raise ValueError(f"No training adapter for {version}")


def _trainer_kwargs(args: argparse.Namespace, device: torch.device, steps: int):
    values = {
        "modelpath": args.modelpath,
        "epochs": args.epochs,
        "lr": args.lr,
        "ecglen": args.ecglen,
        "kernel_size": args.kernel_size,
        "pool_size": args.pool_size,
        "figure_plot": 0,
        "step_total": steps,
        "anylead": args.anylead,
        "patience": args.patience,
        "device": device,
    }
    if args.version not in {"v0", "v1"}:
        values["alpha"] = args.alpha
    if args.version in CURRICULUM_VERSIONS:
        values["stage1_epochs"] = args.stage1_epochs
    if args.version == "v15":
        values["lambda_"] = args.lambda_
    return values


def configure_fixed_beta(trainer, fixed_beta: float | None) -> None:
    if fixed_beta is None:
        return
    gate = getattr(trainer.model, "residual_gate_logit", None)
    if gate is None:
        raise AttributeError("v13 model has no residual_gate_logit")
    logit = math.log(fixed_beta / (0.2 - fixed_beta))
    with torch.no_grad():
        gate.fill_(logit)
    gate.requires_grad_(False)


def _run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": args.version,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "ecglen": args.ecglen,
        "kernel_size": args.kernel_size,
        "pool_size": args.pool_size,
        "anylead": args.anylead,
        "stage1_epochs": args.stage1_epochs,
        "alpha": args.alpha,
        "lambda": args.lambda_,
        "fixed_beta": args.fixed_beta,
        "multirate": args.multirate,
        "multirate_rates_hz": list(MULTIRATE_HZ) if args.multirate else [],
    }


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch"}
    missing = required - state.keys()
    if missing:
        raise ValueError(f"Resume checkpoint is missing RNG states: {sorted(missing)}")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _checkpoint_payload(
    trainer,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, Any]:
    payload = {
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "ecglen": args.ecglen,
        "kernel_size": args.kernel_size,
        "pool_size": args.pool_size,
        "best_loss1": trainer.best_loss1,
        "waiting": trainer.waiting,
        "config": _run_config(args),
        "rng_state": _rng_state(),
    }
    payload.update(checkpoint_metadata_fields(metadata, trainer.optimizer, epoch))
    return payload


def _same_config_value(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return expected == actual


def resume_training(trainer, args: argparse.Namespace) -> int:
    if args.resume is None:
        return 0
    checkpoint = torch.load(args.resume, map_location=trainer.device, weights_only=False)
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("Resume checkpoint has no strict config block")
    current_config = _run_config(args)
    for field in ("version", "seed", "alpha", "lambda", "batch_size", "lr"):
        if not _same_config_value(current_config[field], saved_config.get(field)):
            raise ValueError(
                f"Resume config mismatch for {field}: "
                f"checkpoint={saved_config.get(field)!r}, CLI={current_config[field]!r}"
            )
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    trainer.best_loss1 = float(checkpoint.get("best_loss1", float("inf")))
    trainer.waiting = int(checkpoint.get("waiting", 0))
    _restore_rng_state(checkpoint.get("rng_state", {}))
    start_epoch = int(checkpoint.get("epoch", 0))
    if start_epoch >= args.epochs:
        raise ValueError(
            f"Resume epoch {start_epoch} is not below requested --epochs {args.epochs}"
        )
    return start_epoch


def _iter_batches(loader, maximum: int | None):
    for index, batch in enumerate(loader):
        if maximum is not None and index >= maximum:
            break
        yield batch


def train_epochs(
    trainer,
    adapter,
    train_loader,
    val_loader,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    start_epoch: int,
) -> None:
    best_path = Path(args.modelpath)
    last_path = best_path.with_name("last.pt")
    batch_limit = 2 if args.smoke_test else None

    for epoch in range(start_epoch, args.epochs):
        is_stage1 = adapter.prepare_epoch(trainer, epoch)
        train_loss_sum = 0.0
        train_pcc_sum = 0.0
        train_batches = 0
        selected_rates: list[int] = []
        for batch in _iter_batches(train_loader, batch_limit):
            if args.multirate:
                batch, selected_hz = apply_multirate_batch(batch)
                selected_rates.append(selected_hz)
            loss, pcc = adapter.train_step(trainer, batch, is_stage1)
            train_loss_sum += loss
            train_pcc_sum += pcc
            train_batches += 1

        val_loss_sum = 0.0
        val_pcc_sum = 0.0
        val_batches = 0
        val_values = 0
        for batch in _iter_batches(val_loader, batch_limit):
            loss, pcc = adapter.val_step(trainer, batch, is_stage1)
            val_loss_sum += loss
            val_pcc_sum += pcc
            val_batches += 1
            val_values += batch.size(0) * 12

        if train_batches == 0 or val_batches == 0:
            raise RuntimeError("Training and validation loaders must both contain data")
        train_loss = train_loss_sum / train_batches
        train_pcc = train_pcc_sum / train_batches
        val_loss = val_loss_sum / val_batches
        val_pcc = val_pcc_sum / max(val_values, 1)

        improved = val_loss < trainer.best_loss1
        if improved:
            trainer.best_loss1 = val_loss
        else:
            trainer.waiting += 1

        completed_epoch = epoch + 1
        metadata.update(
            {
                "completed_epoch": completed_epoch,
                "best_val_loss": trainer.best_loss1,
                "last_train_loss": train_loss,
                "last_train_pcc": train_pcc,
                "last_val_loss": val_loss,
                "last_val_pcc": val_pcc,
            }
        )
        payload = _checkpoint_payload(trainer, metadata, args, completed_epoch)
        if improved:
            torch.save(payload, best_path)
        torch.save(payload, last_path)
        write_run_metadata(metadata, metadata_path_for(best_path))

        rate_note = f" rates={selected_rates}" if selected_rates else ""
        print(
            f"epoch={completed_epoch}/{args.epochs} train_loss={train_loss:.6f} "
            f"train_pcc={train_pcc:.6f} val_loss={val_loss:.6f} "
            f"val_pcc={val_pcc:.6f}{rate_note}"
        )
        if trainer.waiting > trainer.patience:
            print("Early stopping")
            break


def main(argv: list[str] | None = None) -> int:
    repo_root = require_repo_root()
    args = validate_and_resolve_args(parse_args(argv))
    for input_path in (args.trainpath, args.valpath):
        if not Path(input_path).is_file():
            raise FileNotFoundError(f"Dataset not found: {Path(input_path).as_posix()}")
    if args.resume is not None and not Path(args.resume).is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume.as_posix()}")

    seed_everything(args.seed)
    module = load_version_module(args.version)
    source_version, source_dirty = resolve_source_version(
        repo_root, explicit=args.source_version
    )
    device = require_cuda_device()

    train_loader = module.make_dataloader(
        args.trainpath,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = module.make_dataloader(
        args.valpath,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    args.modelpath.parent.mkdir(parents=True, exist_ok=True)
    trainer = module.Trainer(
        **_trainer_kwargs(args, device=device, steps=len(train_loader))
    )
    configure_fixed_beta(trainer, args.fixed_beta)
    start_epoch = resume_training(trainer, args)
    configure_fixed_beta(trainer, args.fixed_beta)

    metadata = build_run_metadata(
        args=args,
        version=args.version,
        source_version=source_version,
        source_dirty=source_dirty,
        optimizer=trainer.optimizer.__class__.__name__,
        device=str(device),
    )
    metadata.update(
        {
            "best_checkpoint": args.modelpath.as_posix(),
            "last_checkpoint": args.modelpath.with_name("last.pt").as_posix(),
            "resume_from": "" if args.resume is None else args.resume.as_posix(),
            "start_epoch": start_epoch,
            "completed_epoch": start_epoch,
            "best_val_loss": trainer.best_loss1,
            "smoke_test": args.smoke_test,
            "fixed_beta": "" if args.fixed_beta is None else args.fixed_beta,
            "multirate": args.multirate,
            "multirate_rates_hz": ";".join(map(str, MULTIRATE_HZ)) if args.multirate else "",
        }
    )
    trainer.run_metadata = metadata
    metadata_path = metadata_path_for(args.modelpath)
    write_run_metadata(metadata, metadata_path)
    try:
        train_epochs(
            trainer,
            adapter_for(args.version),
            train_loader,
            val_loader,
            args,
            metadata,
            start_epoch,
        )
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["finished_at_utc"] = _utc_now()
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        write_run_metadata(metadata, metadata_path)
        raise
    metadata["status"] = "success"
    metadata["finished_at_utc"] = _utc_now()
    write_run_metadata(metadata, metadata_path)
    print(f"best: {args.modelpath.as_posix()}")
    print(f"last: {args.modelpath.with_name('last.pt').as_posix()}")
    print(f"metadata: {metadata_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
