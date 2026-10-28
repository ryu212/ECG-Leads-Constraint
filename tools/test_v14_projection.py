"""Run CPU/GPU contract tests for the v14 exact EG projection model."""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, require_repo_root


TEST_NAMES = (
    "import_contract",
    "tensor_shapes",
    "eg_identities_pred12",
    "eg_identities_base12",
    "projection_zero_params",
    "beta_contract",
    "backward_two_heads",
    "curriculum_freeze_unfreeze",
    "save_load",
    "reject_v13_checkpoint",
)

FIELDNAMES = (
    "test_id",
    "version",
    "device",
    "test_name",
    "input_shape",
    "base8_shape",
    "residual8_shape",
    "output8_shape",
    "pred12_shape",
    "base12_shape",
    "err_iii",
    "err_avr",
    "err_avl",
    "err_avf",
    "projection_trainable_params",
    "base_head_grad",
    "residual_head_grad",
    "save_load_match",
    "v13_strict_load_rejected",
    "status",
    "details",
)


def _shape(value) -> str:
    return "x".join(str(part) for part in value.shape)


def _base_row(device: str, test_name: str) -> dict[str, Any]:
    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "test_id": f"v14_{device}_{test_name}",
            "version": "v14",
            "device": device,
            "test_name": test_name,
            "status": "PENDING",
        }
    )
    return row


def _execute(
    rows: list[dict[str, Any]],
    device: str,
    test_name: str,
    operation: Callable[[], dict[str, Any] | None],
) -> None:
    row = _base_row(device, test_name)
    try:
        values = operation() or {}
        row.update(values)
        row["status"] = "PASS"
        row.setdefault("details", "")
    except Exception as exc:
        row["status"] = "FAIL"
        row["details"] = f"{type(exc).__name__}: {exc}"
    rows.append(row)


def _identity_payload(errors: dict[str, Any], tolerance: float = 1e-6):
    values = {name: float(value.detach().cpu()) for name, value in errors.items()}
    maximum = max(values.values())
    if maximum >= tolerance:
        raise AssertionError(
            f"maximum EG identity error {maximum:.9g} is not below {tolerance}"
        )
    return {
        "err_iii": values["III"],
        "err_avr": values["aVR"],
        "err_avl": values["aVL"],
        "err_avf": values["aVF"],
        "details": f"max_abs_error={maximum:.9g}; tolerance={tolerance}",
    }


def _run_device(device_name: str) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    if device_name == "cuda" and not torch.cuda.is_available():
        for test_name in TEST_NAMES:
            row = _base_row(device_name, test_name)
            row["status"] = "SKIP_NO_GPU"
            row["details"] = "CUDA is not available; rerun on a Kaggle GPU session"
            rows.append(row)
        return rows

    device = torch.device(device_name)

    try:
        from tools.train_experiment import CurriculumAdapter
        from v13.model import MCMA_model as V13Model
        from v14.eg_projection import eg_identity_errors
        from v14.model import MCMA_model as V14Model
        from v14.utils_MCMA import Trainer as V14Trainer
    except Exception as exc:
        for test_name in TEST_NAMES:
            row = _base_row(device_name, test_name)
            row["status"] = "FAIL"
            row["details"] = f"import failed: {type(exc).__name__}: {exc}"
            rows.append(row)
        return rows

    _execute(
        rows,
        device_name,
        "import_contract",
        lambda: {
            "details": (
                f"model={V14Model.__module__}.{V14Model.__name__}; "
                f"trainer={V14Trainer.__module__}.{V14Trainer.__name__}"
            )
        },
    )

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    model = V14Model(kernel_size=5).to(device)
    sample = torch.randn(2, 12, 1024, device=device)

    model.eval()
    with torch.no_grad():
        components = model.forward_components(sample)
        public_pred12, public_base12 = model(sample)

    expected_shapes = {
        "base8": (2, 8, 1024),
        "residual8": (2, 8, 1024),
        "output8": (2, 8, 1024),
        "pred12": (2, 12, 1024),
        "base12": (2, 12, 1024),
    }

    def test_shapes():
        for name, expected in expected_shapes.items():
            actual = tuple(components[name].shape)
            if actual != expected:
                raise AssertionError(f"{name}: expected {expected}, got {actual}")
        if not torch.allclose(public_pred12, components["pred12"], atol=1e-7, rtol=1e-6):
            raise AssertionError("public pred12 differs from forward_components pred12")
        if not torch.allclose(public_base12, components["base12"], atol=1e-7, rtol=1e-6):
            raise AssertionError("public base12 differs from forward_components base12")
        if not torch.allclose(
            components["pred12"][:, 6:],
            components["output8"][:, 2:],
            atol=1e-7,
            rtol=1e-6,
        ):
            raise AssertionError("pred12 precordial leads differ from output8 V1..V6")
        if not torch.allclose(
            components["base12"][:, 6:],
            components["base8"][:, 2:],
            atol=1e-7,
            rtol=1e-6,
        ):
            raise AssertionError("base12 precordial leads differ from base8 V1..V6")
        return {
            "input_shape": _shape(sample),
            "base8_shape": _shape(components["base8"]),
            "residual8_shape": _shape(components["residual8"]),
            "output8_shape": _shape(components["output8"]),
            "pred12_shape": _shape(components["pred12"]),
            "base12_shape": _shape(components["base12"]),
            "details": "public tuple and all internal tensor shapes match the v14 contract",
        }

    _execute(rows, device_name, "tensor_shapes", test_shapes)
    _execute(
        rows,
        device_name,
        "eg_identities_pred12",
        lambda: _identity_payload(eg_identity_errors(components["pred12"])),
    )
    _execute(
        rows,
        device_name,
        "eg_identities_base12",
        lambda: _identity_payload(eg_identity_errors(components["base12"])),
    )

    def test_projection_parameters():
        count = sum(parameter.numel() for parameter in model.eg.parameters())
        buffers = dict(model.eg.named_buffers())
        if count != 0:
            raise AssertionError(f"projection has {count} trainable parameters")
        if "P" not in buffers or tuple(buffers["P"].shape) != (12, 8):
            raise AssertionError("projection buffer P is missing or has wrong shape")
        return {
            "projection_trainable_params": count,
            "details": "P is a registered 12x8 buffer and projection has zero parameters",
        }

    _execute(rows, device_name, "projection_zero_params", test_projection_parameters)

    def test_beta():
        beta = float(model.residual_beta.detach().cpu())
        if abs(beta - 0.1) >= 1e-7:
            raise AssertionError(f"expected initial beta 0.1, got {beta}")
        if not model.residual_gate_logit.requires_grad:
            raise AssertionError("residual gate must remain trainable for v14 stage 2")
        return {"details": f"beta={beta:.9g}; trainable=true"}

    _execute(rows, device_name, "beta_contract", test_beta)

    def test_backward():
        model.train()
        model.zero_grad(set_to_none=True)
        current = model.forward_components(sample)
        loss = current["pred12"].pow(2).mean() + current["base12"].pow(2).mean()
        loss.backward()
        base_grad = model.output_head.net[-1].weight.grad
        residual_grad = model.residual_branch[-1].weight.grad
        for name, gradient in (("base", base_grad), ("residual", residual_grad)):
            if gradient is None:
                raise AssertionError(f"{name} head gradient is missing")
            if not torch.isfinite(gradient).all():
                raise AssertionError(f"{name} head gradient contains NaN or Inf")
            if torch.count_nonzero(gradient).item() == 0:
                raise AssertionError(f"{name} head gradient is entirely zero")
        return {
            "base_head_grad": "finite_nonzero",
            "residual_head_grad": "finite_nonzero",
            "details": f"loss={float(loss.detach().cpu()):.9g}",
        }

    _execute(rows, device_name, "backward_two_heads", test_backward)

    def test_curriculum():
        trainer = SimpleNamespace(model=model, stage1_epochs=1)
        stage1 = CurriculumAdapter.prepare_epoch(trainer, 0)
        stage1_frozen = all(
            not parameter.requires_grad
            for parameter in model.residual_branch.parameters()
        )
        stage2 = CurriculumAdapter.prepare_epoch(trainer, 1)
        stage2_trainable = all(
            parameter.requires_grad
            for parameter in model.residual_branch.parameters()
        )
        if not stage1 or not stage1_frozen:
            raise AssertionError("stage 1 did not freeze the residual head")
        if stage2 or not stage2_trainable:
            raise AssertionError("stage 2 did not unfreeze the residual head")
        return {"details": "stage1=frozen; stage2=trainable"}

    _execute(rows, device_name, "curriculum_freeze_unfreeze", test_curriculum)

    def test_save_load():
        model.eval()
        with torch.no_grad():
            before = tuple(value.detach().cpu() for value in model(sample))
        with tempfile.TemporaryDirectory(prefix="v14_save_load_") as temp_dir:
            checkpoint_path = Path(temp_dir) / "state.pt"
            torch.save(model.state_dict(), checkpoint_path)
            restored = V14Model(kernel_size=5).to(device)
            state = torch.load(checkpoint_path, map_location=device)
            restored.load_state_dict(state, strict=True)
            restored.eval()
            with torch.no_grad():
                after = tuple(value.detach().cpu() for value in restored(sample))
            if "eg.P" not in state:
                raise AssertionError("checkpoint state does not contain eg.P")
            if not all(
                torch.allclose(left, right, atol=1e-7, rtol=1e-6)
                for left, right in zip(before, after)
            ):
                raise AssertionError("save-load changed pred12 or base12")
            del restored
        return {"save_load_match": True, "details": "strict v14 save-load preserves both outputs"}

    _execute(rows, device_name, "save_load", test_save_load)

    def test_reject_v13():
        legacy = V13Model(kernel_size=5)
        rejected = False
        try:
            model.load_state_dict(legacy.state_dict(), strict=True)
        except RuntimeError:
            rejected = True
        finally:
            del legacy
        if not rejected:
            raise AssertionError("v13 state_dict unexpectedly strict-loaded into v14")
        return {
            "v13_strict_load_rejected": True,
            "details": "v13 12-channel heads are intentionally incompatible with v14",
        }

    _execute(rows, device_name, "reject_v13_checkpoint", test_reject_v13)

    del components, public_pred12, public_base12, sample, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "all"),
        default="all",
        help="Run one device suite or both CPU and CUDA suites.",
    )
    parser.add_argument(
        "--out",
        type=artifact_path,
        default=Path("artifacts/preflight/v14_unit_tests.csv"),
    )
    return parser.parse_args()


def main() -> int:
    require_repo_root()
    args = parse_args()
    devices = ("cpu", "cuda") if args.device == "all" else (args.device,)
    rows = [row for device in devices for row in _run_device(device)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['status']:11} {row['device']:4} "
            f"{row['test_name']}: {row['details']}"
        )
    print(f"Saved: {args.out.as_posix()}")

    if any(row["status"] == "FAIL" for row in rows):
        return 1
    if any(row["status"] == "SKIP_NO_GPU" for row in rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
