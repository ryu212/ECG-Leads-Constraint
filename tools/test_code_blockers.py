"""Run smoke tests for the four code blockers documented in the runbook."""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, require_repo_root


@dataclass(frozen=True)
class TestResult:
    test_id: str
    target: str
    status: str
    details: str


def _import_module(module_name: str) -> str:
    importlib.import_module(module_name)
    return f"Imported {module_name} from repo root"


def _test_hr_import() -> str:
    return _import_module("v13.evaluating_code.evaluate_model_HR")


def _test_signal_import() -> str:
    return _import_module("v13.evaluating_code.evaluate_signal_MCMA")


def _test_model_output_contract() -> str:
    import torch

    from v13.evaluating_code.evaluate_model_HR import prediction_tensor

    class TensorModel(torch.nn.Module):
        def forward(self, value):
            return value + 1

    class TupleModel(torch.nn.Module):
        def forward(self, value):
            prediction = value + 1
            return prediction, value

    sample = torch.zeros((1, 12, 16), dtype=torch.float32)
    expected = torch.ones_like(sample)

    tensor_prediction = prediction_tensor(TensorModel()(sample))
    tuple_prediction = prediction_tensor(TupleModel()(sample))
    if not torch.equal(tensor_prediction, expected):
        raise AssertionError("Tensor-returning model produced the wrong prediction")
    if not torch.equal(tuple_prediction, expected):
        raise AssertionError("Tuple-returning model produced the wrong prediction")
    return "Tensor and tuple model outputs both resolve to the prediction tensor"


def _test_hdf5_transform_order() -> str:
    import h5py
    import numpy as np
    import torch

    from hdf5_dataset import ECGHDF5Dataset

    source = np.arange(24, dtype=np.float32).reshape(1, 2, 12)
    transform_received_tensor = False

    def transform(ecg):
        nonlocal transform_received_tensor
        transform_received_tensor = isinstance(ecg, torch.Tensor)
        return ecg + 1

    with tempfile.TemporaryDirectory(prefix="mcma_code_blocker_") as temp_dir:
        hdf5_path = Path(temp_dir) / "smoke.hdf5"
        with h5py.File(hdf5_path, "w") as handle:
            handle.create_dataset("tracings", data=source)

        dataset = ECGHDF5Dataset(hdf5_path, transform=transform)
        actual = dataset[0]
        expected = torch.from_numpy(source[0]).float() + 1
        if dataset.file is not None:
            dataset.file.close()
            dataset.file = None

    if not transform_received_tensor:
        raise AssertionError("Transform did not receive an initialized ECG tensor")
    if not torch.equal(actual, expected):
        raise AssertionError("HDF5 transform result does not match the expected tensor")
    return "HDF5 record is converted to a tensor before transform is called"


TESTS = (
    (
        "v13_hr_import",
        "v13/evaluating_code/evaluate_model_HR.py",
        _test_hr_import,
    ),
    (
        "v13_signal_import",
        "v13/evaluating_code/evaluate_signal_MCMA.py",
        _test_signal_import,
    ),
    (
        "model_output_contract",
        "v13/evaluating_code/evaluate_model_HR.py",
        _test_model_output_contract,
    ),
    (
        "hdf5_transform_order",
        "hdf5_dataset.py",
        _test_hdf5_transform_order,
    ),
)


def run_tests() -> list[TestResult]:
    results: list[TestResult] = []
    for test_id, target, test in TESTS:
        try:
            details = test()
            status = "PASS"
        except Exception as exc:
            status = "FAIL"
            details = f"{type(exc).__name__}: {exc}"
        results.append(TestResult(test_id, target, status, details))
    return results


def write_report(results: list[TestResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("test_id", "target", "status", "details"),
        )
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=artifact_path,
        default=Path("artifacts/preflight/code_blocker_tests.csv"),
    )
    return parser.parse_args()


def main() -> int:
    require_repo_root()
    args = parse_args()
    results = run_tests()
    write_report(results, args.out)

    for result in results:
        print(f"{result.status:4} {result.test_id}: {result.details}")
    print(f"Saved: {args.out.as_posix()}")
    return 0 if all(result.status == "PASS" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
