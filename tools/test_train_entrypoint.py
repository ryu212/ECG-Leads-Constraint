"""Run bounded Kaggle smoke tests for the common training entrypoint."""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, require_repo_root


@dataclass(frozen=True)
class TestResult:
    version: str
    experiment: str
    status: str
    details: str


RUNNABLE_CASES = (
    ("v0", "no-wrfb-single-output", 40, ()),
    ("v1", "single-output", 42, ()),
    ("v3", "configurable-alpha", 42, ("--alpha", "0.1")),
    ("v4", "configurable-alpha", 42, ("--alpha", "0.1")),
    ("v7", "configurable-alpha", 42, ("--alpha", "0.1")),
    ("v8", "curriculum-tuple", 42, ("--alpha", "0.5", "--stage1_epochs", "1")),
    ("v12", "instance-norm-curriculum", 42, ("--alpha", "0.5", "--stage1_epochs", "1")),
    ("v13", "curriculum-tuple", 42, ("--alpha", "0.5", "--stage1_epochs", "1")),
    ("v13", "fixed-gate-beta-0.1", 43, ("--alpha", "0.5", "--stage1_epochs", "1", "--fixed_beta", "0.1")),
    ("v13", "multi-rate-e7", 44, ("--alpha", "0.5", "--stage1_epochs", "1", "--multirate")),
    ("v14", "exact-eg-projection", 45, ("--alpha", "0.5", "--stage1_epochs", "1")),
    ("v15", "soft-eg-lambda", 46, ("--alpha", "0.5", "--lambda", "1.0", "--stage1_epochs", "1")),
)


def _write_synthetic_datasets() -> tuple[Path, Path]:
    import torch

    smoke_data = Path("artifacts/smoke/_entrypoint_test")
    smoke_data.mkdir(parents=True, exist_ok=True)
    trainpath = smoke_data / "train.pt"
    valpath = smoke_data / "val.pt"
    generator = torch.Generator().manual_seed(2026)
    torch.save(torch.randn((2, 1024, 12), generator=generator), trainpath)
    torch.save(torch.randn((2, 1024, 12), generator=generator), valpath)
    return trainpath, valpath


def _test_output_contract() -> None:
    import torch

    from tools.train_experiment import prediction_tensor

    value = torch.ones((1, 12, 16))
    if prediction_tensor(value) is not value:
        raise AssertionError("Tensor output was not preserved")
    if prediction_tensor((value, torch.zeros_like(value))) is not value:
        raise AssertionError("Tuple output did not resolve to its prediction")


def _test_multirate_contract() -> None:
    import random

    import torch

    from tools.train_experiment import MULTIRATE_HZ, apply_multirate_batch, parse_args

    base_cli = [
        "--trainpath", "datasets/pt/train.pt",
        "--valpath", "datasets/pt/val.pt",
        "--modelpath", "artifacts/checkpoints/test/best.pt",
        "--version", "v13",
        "--seed", "42",
        "--epochs", "1",
        "--batch_size", "1",
        "--lr", "0.001",
    ]
    if parse_args(base_cli).multirate:
        raise AssertionError("Multi-rate must default to disabled")
    if not parse_args([*base_cli, "--multirate"]).multirate:
        raise AssertionError("--multirate did not enable augmentation")
    if parse_args([*base_cli, "--no-multirate"]).multirate:
        raise AssertionError("--no-multirate did not disable augmentation")

    batch = torch.randn((2, 1024, 12), dtype=torch.float32)
    for rate_hz in MULTIRATE_HZ:
        transformed, selected = apply_multirate_batch(batch, rate_hz=rate_hz)
        if selected != rate_hz or transformed.shape != batch.shape:
            raise AssertionError(f"Shape/rate contract failed at {rate_hz} Hz")
        if transformed.dtype != batch.dtype or not torch.isfinite(transformed).all():
            raise AssertionError(f"dtype/finite contract failed at {rate_hz} Hz")
    random.seed(123)
    first = [apply_multirate_batch(batch)[1] for _ in range(8)]
    random.seed(123)
    second = [apply_multirate_batch(batch)[1] for _ in range(8)]
    if first != second:
        raise AssertionError("Seeded multi-rate choices are not reproducible")


def _run_case(
    version: str,
    experiment: str,
    seed: int,
    extra_args: tuple[str, ...],
    trainpath: Path,
    valpath: Path,
) -> TestResult:
    import torch

    from tools.train_experiment import main as train_main

    argv = [
        "--version",
        version,
        "--trainpath",
        trainpath.as_posix(),
        "--valpath",
        valpath.as_posix(),
        "--modelpath",
        f"artifacts/checkpoints/_entrypoint_test/{version}/best.pt",
        "--seed",
        str(seed),
        "--epochs",
        "1",
        "--batch_size",
        "1",
        "--lr",
        "0.001",
        "--smoke_test",
        "--source-version",
        "entrypoint-smoke",
        *extra_args,
    ]
    try:
        result = train_main(argv)
        best = Path(f"artifacts/smoke/{version}/seed_{seed}/best.pt")
        last = best.with_name("last.pt")
        metadata = best.with_name("run_metadata.csv")
        if result != 0 or not all(path.is_file() for path in (best, last, metadata)):
            raise AssertionError("best.pt, last.pt, or run_metadata.csv is missing")
        return TestResult(
            version,
            experiment,
            "PASS",
            "1 epoch; <=2 train/val batches; best/last/metadata written",
        )
    except Exception as exc:
        return TestResult(version, experiment, "FAIL", f"{type(exc).__name__}: {exc}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_tests() -> list[TestResult]:
    results: list[TestResult] = []
    try:
        _test_output_contract()
        _test_multirate_contract()
    except Exception as exc:
        details = f"Shared contract failed: {type(exc).__name__}: {exc}"
        results.extend(
            TestResult(version, experiment, "FAIL", details)
            for version, experiment, _, _ in RUNNABLE_CASES
        )
    else:
        trainpath, valpath = _write_synthetic_datasets()
        for case in RUNNABLE_CASES:
            results.append(_run_case(*case, trainpath, valpath))

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=artifact_path,
        default=Path("artifacts/preflight/train_entrypoint_tests.csv"),
    )
    return parser.parse_args()


def main() -> int:
    require_repo_root()
    args = parse_args()
    results = run_tests()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("version", "experiment", "status", "details"),
        )
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    for result in results:
        print(f"{result.status:14} {result.version}/{result.experiment}: {result.details}")
    print(f"Saved: {args.out.as_posix()}")
    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
