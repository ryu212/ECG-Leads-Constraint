"""Aggregate E3 per-seed CSVs and render reproducible PNG/PDF figures."""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, require_repo_root


ALPHA_PATTERN = re.compile(r"alpha[_-]?(0(?:[.][0-9]+)?|1(?:[.]0+)?)", re.IGNORECASE)
SEED_PATTERN = re.compile(r"seed[_-]?([0-9]+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument(
        "--metrics-dir", type=artifact_path, default=Path("artifacts/metrics/E3")
    )
    parser.add_argument(
        "--figures-dir", type=artifact_path, default=Path("artifacts/figures/E3")
    )
    return parser.parse_args()


def main() -> int:
    require_repo_root()
    args = parse_args()
    paths = sorted({Path(match) for pattern in args.inputs for match in glob.glob(pattern, recursive=True)})
    if not paths:
        raise FileNotFoundError("No E3 CSV files matched --inputs")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if not {"tag", "lead", "E_nd_true", "E_nd_pred", "ratio_pred_over_true"}.issubset(frame.columns):
            continue
        alpha_match = ALPHA_PATTERN.search(str(frame["tag"].iloc[0]))
        seed_match = SEED_PATTERN.search(str(frame["tag"].iloc[0]))
        if not alpha_match or not seed_match:
            raise ValueError(f"E3 tag must contain alpha and seed: {path}")
        frame["alpha"] = float(alpha_match.group(1))
        frame["seed"] = int(seed_match.group(1))
        frame["source_file"] = path.as_posix()
        frames.append(frame)
    if not frames:
        raise RuntimeError("Matched files did not contain the E3 schema")
    combined = pd.concat(frames, ignore_index=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.metrics_dir / "per_seed.csv", index=False)
    summary = (
        combined.groupby(["alpha", "lead"], as_index=False)[
            ["E_nd_true", "E_nd_pred", "ratio_pred_over_true"]
        ]
        .agg(["mean", "std", "count"])
    )
    summary.columns = ["_".join(value).rstrip("_") for value in summary.columns]
    summary.to_csv(args.metrics_dir / "summary.csv", index=False)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for lead, part in combined.groupby("lead"):
        values = part.groupby("alpha")["ratio_pred_over_true"].agg(["mean", "std"]).reset_index()
        axis.errorbar(values["alpha"], values["mean"], yerr=values["std"].fillna(0), marker="o", capsize=3, label=lead)
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("VCG loss weight alpha")
    axis.set_ylabel("Predicted / true non-dipolar energy")
    axis.grid(alpha=0.25)
    axis.legend(ncol=4, fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(args.figures_dir / f"nondipolar_energy.{suffix}", dpi=300)
    plt.close(figure)
    print(f"Saved: {(args.metrics_dir / 'summary.csv').as_posix()}")
    print(f"Saved figures under: {args.figures_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
