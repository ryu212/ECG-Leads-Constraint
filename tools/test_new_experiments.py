"""Smoke-test v0/v15 architecture, loss, and common training CLI contracts."""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, require_repo_root


@dataclass
class Result:
    device: str
    test: str
    status: str
    details: str


def execute(rows: list[Result], device: str, name: str, function) -> None:
    try:
        details = function()
        rows.append(Result(device, name, "PASS", str(details)))
    except Exception as exc:
        rows.append(Result(device, name, "FAIL", f"{type(exc).__name__}: {exc}"))


def run_device(device: torch.device) -> list[Result]:
    from v0.model import MCMA_model as V0Model
    from v1.model import MCMA_model as V1Model
    from v13.model import MCMA_model as V13Model
    from v15.loss.loss import SoftEGConstraintLoss, soft_eg_penalty
    from v15.model import MCMA_model as V15Model

    rows: list[Result] = []
    device_name = device.type
    v0 = V0Model(kernel_size=5).to(device).eval()
    v1 = V1Model(kernel_size=5).to(device).eval()
    v13 = V13Model(kernel_size=5).to(device).eval()
    v15 = V15Model(kernel_size=5).to(device).eval()
    sample = torch.randn(1, 12, 1024, device=device)

    def test_v0():
        if any("WRFB" in name for name, _ in v0.named_modules()):
            raise AssertionError("v0 still registers a WRFB module")
        v0_params = sum(parameter.numel() for parameter in v0.parameters())
        v1_params = sum(parameter.numel() for parameter in v1.parameters())
        if not v0_params < v1_params:
            raise AssertionError(f"v0 params={v0_params} is not below v1 params={v1_params}")
        with torch.no_grad():
            output = v0(sample)
        if output.shape != sample.shape:
            raise AssertionError(f"v0 output shape is {tuple(output.shape)}")
        return f"shape={tuple(output.shape)}; params={v0_params} < v1={v1_params}"

    execute(rows, device_name, "v0_no_wrfb", test_v0)

    def test_v15_model():
        if set(v13.state_dict()) != set(v15.state_dict()):
            raise AssertionError("v15 architecture/state keys differ from v13")
        with torch.no_grad():
            pred, base = v15(sample)
        if pred.shape != sample.shape or base.shape != sample.shape:
            raise AssertionError("v15 tuple output shape is invalid")
        return "v13-compatible keys; tuple outputs are (B,12,L)"

    execute(rows, device_name, "v15_architecture", test_v15_model)

    def test_soft_penalty():
        lead_i = torch.randn(2, 1024, device=device)
        lead_ii = torch.randn(2, 1024, device=device)
        exact = torch.randn(2, 12, 1024, device=device)
        exact[:, 0], exact[:, 1] = lead_i, lead_ii
        exact[:, 2] = lead_ii - lead_i
        exact[:, 3] = -0.5 * (lead_i + lead_ii)
        exact[:, 4] = lead_i - 0.5 * lead_ii
        exact[:, 5] = lead_ii - 0.5 * lead_i
        exact_value = soft_eg_penalty(exact).item()
        random_value = soft_eg_penalty(torch.randn_like(exact)).item()
        if exact_value > 1e-12 or random_value <= 0:
            raise AssertionError(f"exact={exact_value}; random={random_value}")
        return f"exact={exact_value:.3g}; random={random_value:.6g}"

    execute(rows, device_name, "v15_soft_eg_penalty", test_soft_penalty)

    def test_lambda_scaling():
        pred = torch.randn(2, 12, 64, device=device, requires_grad=True)
        true = torch.randn_like(pred)
        base = torch.randn_like(pred)
        loss0 = SoftEGConstraintLoss(alpha=0.5, lambda_=0.0, device=device)(pred, true, base)
        loss2 = SoftEGConstraintLoss(alpha=0.5, lambda_=2.0, device=device)(pred, true, base)
        expected = 2.0 * soft_eg_penalty(pred)
        if not torch.allclose(loss2 - loss0, expected, atol=1e-6, rtol=1e-5):
            raise AssertionError("lambda does not scale only the soft EG term")
        loss2.backward()
        if pred.grad is None or not torch.isfinite(pred.grad).all():
            raise AssertionError("soft loss backward failed")
        return f"loss_delta={(loss2-loss0).item():.6g}"

    execute(rows, device_name, "v15_lambda_scaling_backward", test_lambda_scaling)
    del v0, v1, v13, v15, sample
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def cli_contract() -> str:
    from tools.train_experiment import parse_args, validate_and_resolve_args

    base = [
        "--trainpath", "datasets/pt/train.pt",
        "--valpath", "datasets/pt/val.pt",
        "--modelpath", "artifacts/checkpoints/test/best.pt",
        "--version", "v15",
        "--seed", "42",
        "--epochs", "100",
        "--batch_size", "256",
        "--lr", "0.001",
        "--alpha", "0.5",
        "--stage1_epochs", "50",
    ]
    try:
        validate_and_resolve_args(parse_args(base))
    except ValueError as exc:
        if "requires --lambda" not in str(exc):
            raise
    else:
        raise AssertionError("v15 accepted a missing --lambda")
    args = validate_and_resolve_args(parse_args([*base, "--lambda", "1.0"]))
    if args.lambda_ != 1.0:
        raise AssertionError("v15 lambda was not preserved")
    return "v15 requires and preserves runtime --lambda"


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=artifact_path,
        default=Path("artifacts/preflight/new_experiment_tests.csv"),
    )
    return parser.parse_args()


def main() -> int:
    require_repo_root()
    args = parse_cli()
    rows: list[Result] = []
    execute(rows, "shared", "v15_cli_lambda", cli_contract)
    rows.extend(run_device(torch.device("cpu")))
    if torch.cuda.is_available():
        rows.extend(run_device(torch.device("cuda")))
    else:
        rows.append(Result("cuda", "runtime", "SKIP", "CUDA is unavailable"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("device", "test", "status", "details"))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    for row in rows:
        print(f"{row.status:5} {row.device:6} {row.test}: {row.details}")
    print(f"Saved: {args.out.as_posix()}")
    return 1 if any(row.status == "FAIL" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
