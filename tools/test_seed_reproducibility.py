"""Smoke-test RNG seeding and the training metadata contracts."""

from __future__ import annotations

import argparse
import csv
import random
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, require_repo_root


@dataclass(frozen=True)
class TestResult:
    test_id: str
    status: str
    details: str


def _random_samples(seed: int):
    import numpy as np
    import torch

    from experiment_metadata import seed_everything

    seed_everything(seed)
    return random.random(), np.random.random(), torch.rand(4)


def _test_python_numpy_torch() -> str:
    import torch

    first = _random_samples(12345)
    second = _random_samples(12345)
    if first[0] != second[0] or first[1] != second[1]:
        raise AssertionError("Python or NumPy RNG did not repeat")
    if not torch.equal(first[2], second[2]):
        raise AssertionError("Torch CPU RNG did not repeat")
    different = _random_samples(54321)
    if first[0] == different[0] or torch.equal(first[2], different[2]):
        raise AssertionError("Different seeds produced identical samples")
    return "Python NumPy and Torch CPU repeat for the same seed"


def _test_cuda() -> str:
    import torch

    from experiment_metadata import seed_everything

    seed_everything(12345)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run this smoke test in a Kaggle GPU session")
    first = torch.rand(4, device="cuda").cpu()
    seed_everything(12345)
    second = torch.rand(4, device="cuda").cpu()
    if not torch.equal(first, second):
        raise AssertionError("Torch CUDA RNG did not repeat")
    return "Torch CUDA repeats for the same seed"


def _test_checkpoint_schema() -> str:
    import torch

    from experiment_metadata import checkpoint_metadata_fields

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metadata = {
        "run_id": "v13-seed42-smoke",
        "seed": 42,
        "version": "v13",
        "epochs": 1,
        "batch_size": 2,
        "lr": 1e-3,
        "ecglen": 16,
        "kernel_size": 5,
        "pool_size": 2,
        "anylead": 1,
        "patience": 1,
        "figure_plot": 0,
        "stage1_epochs": 1,
        "alpha": 0.5,
        "lambda": "",
        "source_version": "smoke-source",
        "source_dirty": False,
    }
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    payload.update(checkpoint_metadata_fields(metadata, optimizer, epoch=1))
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "seed",
        "version",
        "hyperparameters",
        "optimizer",
        "source_version",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise AssertionError(f"Missing checkpoint keys: {missing}")

    with tempfile.TemporaryDirectory(prefix="mcma_seed_test_") as temp_dir:
        checkpoint_path = Path(temp_dir) / "checkpoint.pt"
        torch.save(payload, checkpoint_path)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if loaded["seed"] != 42 or loaded["epoch"] != 1:
        raise AssertionError("Checkpoint metadata did not round-trip")
    return "Checkpoint metadata schema saves and loads successfully"


def _test_run_metadata_schema() -> str:
    from experiment_metadata import (
        RUN_METADATA_COLUMNS,
        build_run_metadata,
        write_run_metadata,
    )

    args = SimpleNamespace(
        seed=42,
        trainpath=Path("datasets/pt/train.pt"),
        valpath=Path("datasets/pt/val.pt"),
        modelpath=Path("artifacts/checkpoints/smoke/best.pt"),
        epochs=1,
        batch_size=2,
        lr=1e-3,
        ecglen=16,
        kernel_size=5,
        pool_size=2,
        anylead=1,
        patience=1,
        figure_plot=0,
        stage1_epochs=1,
        alpha=0.5,
    )
    metadata = build_run_metadata(
        args=args,
        version="v13",
        source_version="smoke-source",
        source_dirty=False,
        optimizer="Adam",
        device="cpu",
    )
    with tempfile.TemporaryDirectory(prefix="mcma_metadata_test_") as temp_dir:
        metadata_path = Path(temp_dir) / "run_metadata.csv"
        write_run_metadata(metadata, metadata_path)
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = tuple(reader.fieldnames or ())
    if columns != RUN_METADATA_COLUMNS or len(rows) != 1:
        raise AssertionError("run_metadata.csv does not match the required schema")
    if rows[0]["seed"] != "42" or rows[0]["lambda"] != "":
        raise AssertionError("run_metadata.csv contains incorrect seed/lambda values")
    return "run_metadata.csv has the stable schema and one run row"


TESTS = (
    ("python_numpy_torch_seed", _test_python_numpy_torch),
    ("cuda_seed", _test_cuda),
    ("checkpoint_metadata", _test_checkpoint_schema),
    ("run_metadata_schema", _test_run_metadata_schema),
)


def run_tests() -> list[TestResult]:
    results: list[TestResult] = []
    for test_id, test in TESTS:
        try:
            details = test()
            status = "PASS"
        except Exception as exc:
            status = "FAIL"
            details = f"{type(exc).__name__}: {exc}"
        results.append(TestResult(test_id, status, details))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=artifact_path,
        default=Path("artifacts/preflight/seed_reproducibility_test.csv"),
    )
    return parser.parse_args()


def main() -> int:
    require_repo_root()
    args = parse_args()
    results = run_tests()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("test_id", "status", "details"))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    for result in results:
        print(f"{result.status:4} {result.test_id}: {result.details}")
    print(f"Saved: {args.out.as_posix()}")
    return 0 if all(result.status == "PASS" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
