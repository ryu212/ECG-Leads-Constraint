"""Measure parameters and MACs/FLOPs with one convention for MCMA versions."""

from __future__ import annotations

import argparse
import importlib
import io
import platform
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, require_repo_root
from tools.experiment_common import utc_now, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", nargs="+", default=["v13", "v14"])
    parser.add_argument("--ecglen", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument(
        "--out-dir", type=artifact_path, default=Path("artifacts/metrics/E1c")
    )
    return parser.parse_args()


def main() -> int:
    require_repo_root()
    args = parse_args()
    if args.ecglen <= 0 or args.channels != 12:
        raise ValueError("Expected a positive ECG length and exactly 12 input channels")
    try:
        import ptflops
        from ptflops import get_model_complexity_info
    except ImportError as exc:
        raise RuntimeError("Install the pinned ptflops dependency before E1c") from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = io.StringIO()
    rows = []
    with redirect_stdout(raw), redirect_stderr(raw):
        print(f"created_at_utc={utc_now()}")
        print(f"python={platform.python_version()}")
        print(f"torch={torch.__version__}")
        print(f"ptflops={getattr(ptflops, '__version__', 'unknown')}")
        print("convention=FLOPs are reported as 2 * MACs")
        for version in args.versions:
            module = importlib.import_module(f"{version}.model")
            model = module.MCMA_model(
                input_size=(args.ecglen, args.channels),
                kernel_size=args.kernel_size,
            ).cpu().eval()
            total = sum(parameter.numel() for parameter in model.parameters())
            trainable = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            macs, ptflops_params = get_model_complexity_info(
                model,
                (args.channels, args.ecglen),
                as_strings=False,
                print_per_layer_stat=True,
                verbose=True,
            )
            projection_params = 0
            projection_buffer_elements = 0
            projection = getattr(model, "eg", None)
            if projection is not None:
                projection_params = sum(parameter.numel() for parameter in projection.parameters())
                projection_buffer_elements = sum(buffer.numel() for buffer in projection.buffers())
            print(f"version={version} macs={macs} params={total}")
            rows.append(
                {
                    "version": version,
                    "input_shape": f"1x{args.channels}x{args.ecglen}",
                    "macs": int(macs),
                    "flops_2x_macs": int(2 * macs),
                    "total_params": int(total),
                    "trainable_params": int(trainable),
                    "ptflops_params": int(ptflops_params),
                    "projection_trainable_params": int(projection_params),
                    "projection_buffer_elements": int(projection_buffer_elements),
                    "flops_convention": "2*MACs",
                    "torch_version": torch.__version__,
                    "ptflops_version": getattr(ptflops, "__version__", "unknown"),
                }
            )
    (args.out_dir / "model_complexity_raw.log").write_text(raw.getvalue(), encoding="utf-8")
    write_csv(args.out_dir / "model_complexity.csv", rows)
    print(f"Saved: {(args.out_dir / 'model_complexity.csv').as_posix()}")
    print(f"Saved: {(args.out_dir / 'model_complexity_raw.log').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

