"""Audit required experiment outputs, write a checksum manifest, and create a ZIP."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import sys
import zipfile
from pathlib import Path


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, require_repo_root
from tools.experiment_common import sha256, utc_now, write_csv


DEFAULT_REQUIREMENTS = (
    ("metrics/E0/v1_ptbxl.csv", 1),
    ("metrics/E0/v1_ptbxl.npz", 1),
    ("metrics/E0/v1_cpsc2018.csv", 1),
    ("metrics/E0/v1_cpsc2018.npz", 1),
    ("metrics/E0/v13_ptbxl.csv", 1),
    ("metrics/E0/v13_ptbxl.npz", 1),
    ("metrics/E0/v13_cpsc2018.csv", 1),
    ("metrics/E0/v13_cpsc2018.npz", 1),
    ("metrics/E1a/lambda_selection.csv", 1),
    ("metrics/E1a/per_seed_metrics.csv", 1),
    ("metrics/E1a/summary.csv", 1),
    ("metrics/E1b/per_seed_metrics.csv", 1),
    ("metrics/E1b/summary.csv", 1),
    ("metrics/E1b/reconstruction/*_identity_check.csv", 1),
    ("metrics/E1c/model_complexity.csv", 1),
    ("metrics/E1c/model_complexity_raw.log", 1),
    ("metrics/E2/reconstruction/*_summary.csv", 1),
    ("generated_ecg/E2/generation_manifest.csv", 1),
    ("predictions/E2/*.npy", 1),
    ("metrics/E2/diagnostic/macro_scores.csv", 1),
    ("metrics/E2/diagnostic/per_seed_metrics.csv", 1),
    ("metrics/E2/diagnostic/summary.csv", 1),
    ("metrics/E3/per_seed.csv", 1),
    ("metrics/E3/summary.csv", 1),
    ("metrics/E3/*.npz", 1),
    ("figures/E3/*.png", 1),
    ("figures/E3/*.pdf", 1),
    ("metrics/E4/summary.csv", 1),
    ("metrics/E4/per_transform_metrics.csv", 1),
    ("predictions/E4/*.npy", 1),
    ("metrics/E5/summary.csv", 1),
    ("metrics/E5/per_config_metrics.csv", 1),
    ("predictions/E5/*.npy", 1),
    ("metrics/E6/ablation_summary.csv", 1),
    ("metrics/E6/wilcoxon.csv", 1),
    ("metrics/E6/**/*_per_record.csv", 1),
    ("metrics/E7/per_rate_metrics.csv", 1),
    ("metrics/E7/summary.csv", 1),
    ("metrics/E8/beat_level_metrics.csv", 1),
    ("metrics/E8/regression_summary.csv", 1),
    ("metrics/E8/bland_altman_summary.csv", 1),
    ("figures/E8/*.png", 1),
    ("figures/E8/*.pdf", 1),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=artifact_path, default=Path("artifacts"))
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--manifest", type=artifact_path, default=Path("artifacts/manifests/final_manifest.csv")
    )
    parser.add_argument(
        "--summary", type=artifact_path, default=Path("artifacts/metrics/summary/experiment_summary.csv")
    )
    parser.add_argument(
        "--zip", dest="zip_path", type=artifact_path, default=Path("artifacts/artifacts_final.zip")
    )
    return parser.parse_args()


def included(path: Path, root: Path, zip_path: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    if path.resolve() == zip_path.resolve():
        return False
    if relative.startswith("smoke/") or "/smoke/" in relative:
        return False
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return False
    if path.name.endswith(".tmp") or path.name == ".gitkeep":
        return False
    return True


def required_audit(root: Path, requirements: list[tuple[str, int]]) -> list[dict]:
    files = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]
    rows = []
    for pattern, expected_minimum in requirements:
        matches = [value for value in files if fnmatch.fnmatch(value, pattern)]
        rows.append(
            {
                "pattern": pattern,
                "expected_minimum": expected_minimum,
                "actual_count": len(matches),
                "status": "PASS" if len(matches) >= expected_minimum else "MISSING",
                "matches": ";".join(matches),
            }
        )
    return rows


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def combined_summary_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    metrics_root = root / "metrics"
    if not metrics_root.is_dir():
        return rows
    candidates = []
    for path in metrics_root.rglob("*.csv"):
        relative = path.relative_to(metrics_root)
        if relative.parts and relative.parts[0] == "summary":
            continue
        name = path.name.lower()
        if any(token in name for token in ("summary", "metrics", "macro_scores", "model_complexity", "lambda_selection")):
            candidates.append(path)
    for path in sorted(candidates):
        relative = path.relative_to(metrics_root)
        experiment = relative.parts[0] if relative.parts else ""
        for row_index, row in enumerate(read_csv_rows(path)):
            rows.append(
                {
                    "_experiment": experiment,
                    "_source_file": path.relative_to(REPOSITORY_DIR).as_posix(),
                    "_source_row": row_index,
                    **row,
                }
            )
    return rows


def union_fieldnames(rows: list[dict], preferred: tuple[str, ...] = ()) -> list[str]:
    names = list(preferred)
    seen = set(names)
    for row in rows:
        for name in row:
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


def combined_statistical_rows(root: Path, target: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((root / "metrics").rglob("*.csv")):
        if path.resolve() == target.resolve():
            continue
        name = path.name.lower()
        if "wilcoxon" not in name and "statistical" not in name:
            continue
        for row in read_csv_rows(path):
            rows.append({"_source_file": path.relative_to(REPOSITORY_DIR).as_posix(), **row})
    return rows


def main() -> int:
    require_repo_root()
    args = parse_args()
    root = args.artifacts_root
    if not root.is_dir():
        raise FileNotFoundError(f"Artifacts root not found: {root}")
    requirements = (
        [(pattern, 1) for pattern in args.require]
        if args.require
        else list(DEFAULT_REQUIREMENTS)
    )
    audit = required_audit(root, requirements)
    missing = [row["pattern"] for row in audit if row["status"] != "PASS"]
    audit_path = root / "manifests" / "final_completeness_check.csv"
    write_csv(audit_path, audit)
    if missing and not args.allow_incomplete:
        raise RuntimeError(f"Required outputs are missing: {missing}; see {audit_path}")

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = combined_summary_rows(root)
    write_csv(
        args.summary,
        summary_rows,
        fieldnames=union_fieldnames(summary_rows, ("_experiment", "_source_file", "_source_row")),
    )
    statistical_path = root / "metrics" / "summary" / "statistical_tests.csv"
    statistical_rows = combined_statistical_rows(root, statistical_path)
    write_csv(
        statistical_path,
        statistical_rows,
        fieldnames=union_fieldnames(statistical_rows, ("_source_file",)),
    )

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and included(path, root, args.zip_path)
        and path.resolve() != args.manifest.resolve()
    )
    manifest_rows = [
        {
            "created_at_utc": utc_now(),
            "path": path.relative_to(REPOSITORY_DIR).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    write_csv(args.manifest, manifest_rows)

    # Re-read after writing so the final manifest itself is included in the ZIP.
    files = sorted(path for path in root.rglob("*") if path.is_file() and included(path, root, args.zip_path))
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(root).as_posix())
    print(f"Packaged {len(files)} files")
    print(f"Saved: {args.manifest.as_posix()}")
    print(f"Saved: {args.zip_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
