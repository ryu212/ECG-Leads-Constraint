"""Audit, prepare, and verify a traceable Kaggle dataset snapshot.

This utility deliberately does not create Kaggle metadata and does not upload
anything.  It creates a clean WRFB_MCMA staging tree and a SHA256 manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT_NAME = "WRFB_MCMA"
MANIFEST_RELATIVE_PATH = Path("artifacts/manifests/source_manifest.csv")
AUDIT_RELATIVE_PATH = Path("artifacts/manifests/preparation_audit.json")
PLAN_RELATIVE_PATH = Path("artifacts/manifests/staging_plan.csv")
SUMMARY_RELATIVE_PATH = Path("artifacts/manifests/preparation_summary.json")
SOURCE_CONFIGURATION_FILES: frozenset[Path] = frozenset(
    Path("artifacts/manifests") / name
    for name in (
        "architecture_decision_v14.csv",
        "seed_training_plan.csv",
        "seed_run_status.csv",
        "e6_ablation_plan.csv",
        "experiment_tool_map.csv",
    )
)

REQUIRED_FILES: tuple[Path, ...] = tuple(
    Path(path)
    for path in (
        "paper_v2/RUNBOOK_CHECKLIST.csv",
        "datasets/pt/ptbxl_trainset.pt",
        "datasets/pt/ptbxl_valset.pt",
        "datasets/pt/ptbxl_testset.pt",
        "datasets/pt/cpsc2018.pt",
        "checkpoints/v1/Autoencoder_ks_5.pt",
        "checkpoints/v3/Autoencoder_ks_5.pt",
        "checkpoints/v4/Autoencoder_ks_5.pt",
        "checkpoints/v7/Autoencoder_ks_5.pt",
        "checkpoints/v8/Autoencoder_ks_5.pt",
        "checkpoints/v13/Autoencoder_ks_5.pt",
        "external/model.hdf5",
        "external/code_test/ecg_tracings.hdf5",
        "external/code_test/gold_standard.csv",
        "external/automatic-ecg-diagnosis/README.md",
    )
)

# CODE-test labels currently live in the bundled classifier repository.  The
# snapshot exposes a second, canonical path beside ecg_tracings.hdf5 without
# changing the user's source tree.
NORMALIZED_ASSET_COPIES: dict[Path, Path] = {
    Path("external/code_test/gold_standard.csv"): Path(
        "external/automatic-ecg-diagnosis/data/annotations/gold_standard.csv"
    )
}
SOURCE_REQUIRED_FILES: tuple[Path, ...] = tuple(
    sorted(
        (set(REQUIRED_FILES) - set(NORMALIZED_ASSET_COPIES))
        | set(NORMALIZED_ASSET_COPIES.values()),
        key=lambda path: path.as_posix(),
    )
)

EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
EXCLUDED_ROOT_DIRECTORIES = {"tmp"}
EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo", ".tmp", ".temp", ".log", ".bak", ".swp"}
EXCLUDED_FILE_NAMES = {".DS_Store", "Thumbs.db"}


class PreparationError(RuntimeError):
    """Raised when an audit, preparation, or verification check fails."""


@dataclass(frozen=True)
class PlanRecord:
    path: str
    size_bytes: int
    kind: str
    action: str
    reason: str


@dataclass(frozen=True)
class ManifestRecord:
    path: str
    size_bytes: int
    sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_relative(path: Path) -> str:
    value = path.as_posix()
    if value.startswith("/") or value == ".." or value.startswith("../"):
        raise PreparationError(f"Unsafe relative path: {value}")
    return value


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_project_root(root: Path) -> Path:
    root = root.resolve()
    if not root.is_dir():
        raise PreparationError(f"Project root is not a directory: {root}")
    if root.name != PROJECT_ROOT_NAME:
        raise PreparationError(
            f"Project root must be named {PROJECT_ROOT_NAME!r}, got {root.name!r}"
        )
    return root


def _validate_required_files(
    root: Path, required_files: Sequence[Path]
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    failures: list[str] = []

    for relative in required_files:
        candidate = root / relative
        status = "PASS"
        detail = ""
        size_bytes = 0
        try:
            if not candidate.exists():
                status, detail = "FAIL", "missing"
            elif not candidate.is_file():
                status, detail = "FAIL", "not_a_file"
            else:
                size_bytes = candidate.stat().st_size
                if size_bytes <= 0:
                    status, detail = "FAIL", "empty"
                else:
                    with candidate.open("rb") as handle:
                        handle.read(1)
        except OSError as exc:
            status, detail = "FAIL", f"unreadable: {exc}"

        result = {
            "path": _normalise_relative(relative),
            "size_bytes": size_bytes,
            "status": status,
            "detail": detail,
        }
        results.append(result)
        if status != "PASS":
            failures.append(f"{result['path']} ({detail})")

    if failures:
        raise PreparationError("Required asset validation failed: " + "; ".join(failures))
    return results


def _excluded_file_reason(path: Path) -> str | None:
    if path.name.startswith("~$"):
        return "office_lock"
    if path.name in EXCLUDED_FILE_NAMES:
        return "platform_cache"
    if path.suffix.lower() in EXCLUDED_FILE_SUFFIXES:
        return "temporary_or_cache_file"
    return None


def _validate_symlink(source_root: Path, path: Path) -> None:
    if not path.is_symlink():
        return
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PreparationError(f"Broken symlink: {path}: {exc}") from exc
    if not _is_relative_to(resolved, source_root):
        raise PreparationError(f"Symlink escapes project root: {path} -> {resolved}")


def collect_plan(source_root: Path) -> list[PlanRecord]:
    """Return a deterministic include/exclude plan without modifying source."""
    source_root = _validate_project_root(source_root)
    records: list[PlanRecord] = []

    for current_text, directory_names, file_names in os.walk(source_root):
        current = Path(current_text)
        current_relative = current.relative_to(source_root)

        kept_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = current / name
            relative = candidate.relative_to(source_root)
            _validate_symlink(source_root, candidate)
            reason: str | None = None
            if name in EXCLUDED_DIRECTORY_NAMES:
                reason = "excluded_directory"
            elif relative.parts and relative.parts[0] in EXCLUDED_ROOT_DIRECTORIES:
                reason = "generated_artifacts"
            elif relative.parts and relative.parts[0] == "artifacts" and relative not in {
                Path("artifacts"),
                Path("artifacts/manifests"),
            }:
                reason = "generated_artifacts"

            if reason:
                records.append(
                    PlanRecord(
                        path=_normalise_relative(relative) + "/",
                        size_bytes=0,
                        kind="directory",
                        action="exclude",
                        reason=reason,
                    )
                )
            else:
                kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in sorted(file_names):
            source = current / name
            relative = source.relative_to(source_root)
            _validate_symlink(source_root, source)
            if not source.is_file():
                raise PreparationError(f"Non-regular file in source tree: {source}")
            reason = _excluded_file_reason(source)
            if relative.parts and relative.parts[0] == "artifacts":
                if relative in SOURCE_CONFIGURATION_FILES:
                    reason = None
                else:
                    reason = "generated_artifacts"
            if reason:
                action = "exclude"
            elif relative in SOURCE_CONFIGURATION_FILES:
                action, reason = "include", "source_configuration"
            elif relative in set(NORMALIZED_ASSET_COPIES.values()):
                action, reason = "exclude", "replaced_by_normalized_asset"
            elif relative in REQUIRED_FILES:
                action, reason = "include", "required_asset"
            else:
                action, reason = "include", "source_snapshot"
            records.append(
                PlanRecord(
                    path=_normalise_relative(relative),
                    size_bytes=source.stat().st_size,
                    kind="file",
                    action=action,
                    reason=reason,
                )
            )

    physical_paths = {record.path for record in records if record.kind == "file"}
    for destination, source in NORMALIZED_ASSET_COPIES.items():
        destination_text = _normalise_relative(destination)
        if destination_text in physical_paths:
            continue
        source_path = source_root / source
        if not source_path.is_file():
            raise PreparationError(
                f"Source for normalized asset is missing: {source_path}"
            )
        records.append(
            PlanRecord(
                path=destination_text,
                size_bytes=source_path.stat().st_size,
                kind="file",
                action="include",
                reason=f"normalized_copy_from:{_normalise_relative(source)}",
            )
        )

    return sorted(records, key=lambda item: item.path)


def _git_state(source_root: Path) -> dict[str, object]:
    state: dict[str, object] = {"head": None, "dirty": None}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        state = {"head": head, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        pass
    return state


def _write_plan(path: Path, records: Sequence[PlanRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "size_bytes", "kind", "action", "reason"),
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def audit_source(source_root: Path) -> dict[str, object]:
    source_root = _validate_project_root(source_root)
    required = _validate_required_files(source_root, SOURCE_REQUIRED_FILES)
    records = collect_plan(source_root)
    included = [r for r in records if r.kind == "file" and r.action == "include"]
    excluded = [r for r in records if r.action == "exclude"]
    payload: dict[str, object] = {
        "generated_at_utc": _utc_now(),
        "source_root": str(source_root),
        "git": _git_state(source_root),
        "included_file_count": len(included),
        "included_size_bytes": sum(record.size_bytes for record in included),
        "excluded_entry_count": len(excluded),
        "required_assets": required,
        "status": "PASS",
    }
    _write_plan(source_root / PLAN_RELATIVE_PATH, records)
    _write_json(source_root / AUDIT_RELATIVE_PATH, payload)
    return payload


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _iter_snapshot_payload(snapshot_root: Path) -> Iterable[Path]:
    for path in snapshot_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(snapshot_root)
        if relative == MANIFEST_RELATIVE_PATH:
            continue
        yield path


def _write_manifest(snapshot_root: Path) -> tuple[list[ManifestRecord], str]:
    manifest_path = snapshot_root / MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        _iter_snapshot_payload(snapshot_root),
        key=lambda path: _normalise_relative(path.relative_to(snapshot_root)),
    )

    records: list[ManifestRecord] = []
    for path in paths:
        relative = path.relative_to(snapshot_root)
        records.append(
            ManifestRecord(
                path=_normalise_relative(relative),
                size_bytes=path.stat().st_size,
                sha256=_sha256(path),
            )
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "size_bytes", "sha256"),
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    return records, _sha256(manifest_path)

def _read_manifest(snapshot_root: Path) -> list[ManifestRecord]:
    manifest_path = snapshot_root / MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        raise PreparationError(f"Manifest is missing: {manifest_path}")
    records: list[ManifestRecord] = []
    seen: set[str] = set()
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "size_bytes", "sha256"]:
            raise PreparationError(f"Unexpected manifest schema: {reader.fieldnames}")
        for row in reader:
            relative = row["path"]
            if relative in seen:
                raise PreparationError(f"Duplicate manifest path: {relative}")
            candidate = Path(relative)
            if candidate.is_absolute() or ".." in candidate.parts or "\\" in relative:
                raise PreparationError(f"Unsafe or non-normalised manifest path: {relative}")
            seen.add(relative)
            try:
                size_bytes = int(row["size_bytes"])
            except ValueError as exc:
                raise PreparationError(f"Invalid size for {relative}") from exc
            sha256 = row["sha256"].lower()
            if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
                raise PreparationError(f"Invalid SHA256 for {relative}")
            records.append(ManifestRecord(relative, size_bytes, sha256))
    if [r.path for r in records] != sorted(r.path for r in records):
        raise PreparationError("Manifest paths are not sorted")
    return records


def verify_snapshot(snapshot_root: Path) -> dict[str, object]:
    snapshot_root = _validate_project_root(snapshot_root)
    required = _validate_required_files(snapshot_root, REQUIRED_FILES)
    records = _read_manifest(snapshot_root)
    expected = {record.path: record for record in records}
    actual = {
        _normalise_relative(path.relative_to(snapshot_root)): path
        for path in _iter_snapshot_payload(snapshot_root)
    }
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise PreparationError(
            f"Manifest file set mismatch; missing={missing}, unexpected={unexpected}"
        )

    failures: list[str] = []
    for relative in sorted(expected):
        record = expected[relative]
        path = actual[relative]
        if path.stat().st_size != record.size_bytes:
            failures.append(f"{relative}: size mismatch")
            continue
        digest = _sha256(path)
        if digest != record.sha256:
            failures.append(f"{relative}: SHA256 mismatch")
    if failures:
        raise PreparationError("Snapshot verification failed: " + "; ".join(failures))

    forbidden: list[str] = []
    for relative, path in actual.items():
        if _excluded_file_reason(path):
            forbidden.append(relative)
        parts = Path(relative).parts
        if any(part in EXCLUDED_DIRECTORY_NAMES for part in parts):
            forbidden.append(relative)
        if parts and parts[0] in EXCLUDED_ROOT_DIRECTORIES:
            forbidden.append(relative)
    if forbidden:
        raise PreparationError(f"Excluded files found in snapshot: {sorted(set(forbidden))}")

    manifest_path = snapshot_root / MANIFEST_RELATIVE_PATH
    return {
        "verified_at_utc": _utc_now(),
        "snapshot_root": str(snapshot_root),
        "file_count": len(records),
        "total_size_bytes": sum(record.size_bytes for record in records),
        "manifest_sha256": _sha256(manifest_path),
        "required_assets": required,
        "unexpected_file_count": 0,
        "status": "PASS",
    }


def prepare_snapshot(source_root: Path, package_root: Path) -> dict[str, object]:
    source_root = _validate_project_root(source_root)
    package_root = package_root.resolve()
    if _is_relative_to(package_root, source_root) or _is_relative_to(source_root, package_root):
        raise PreparationError(
            "Package root and source root must not contain one another: "
            f"source={source_root}, package={package_root}"
        )
    snapshot_root = package_root / PROJECT_ROOT_NAME
    if snapshot_root.exists() or snapshot_root.is_symlink():
        raise PreparationError(f"Snapshot already exists; refusing to overwrite: {snapshot_root}")

    audit = audit_source(source_root)
    records = collect_plan(source_root)
    package_root.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir()

    for record in records:
        if record.kind != "file" or record.action != "include":
            continue
        relative = Path(record.path)
        source = source_root / relative
        if not source.exists() and relative in NORMALIZED_ASSET_COPIES:
            source = source_root / NORMALIZED_ASSET_COPIES[relative]
        destination = snapshot_root / Path(record.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=True)

    manifest_records, manifest_sha256 = _write_manifest(snapshot_root)
    source_manifest = source_root / MANIFEST_RELATIVE_PATH
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot_root / MANIFEST_RELATIVE_PATH, source_manifest)

    verification = verify_snapshot(snapshot_root)
    summary: dict[str, object] = {
        "generated_at_utc": _utc_now(),
        "source_root": str(source_root),
        "package_root": str(package_root),
        "snapshot_root": str(snapshot_root),
        "git": audit["git"],
        "file_count": len(manifest_records),
        "total_size_bytes": sum(record.size_bytes for record in manifest_records),
        "manifest_sha256": manifest_sha256,
        "verification": verification["status"],
        "status": "PASS",
    }
    _write_json(source_root / SUMMARY_RELATIVE_PATH, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Audit without copying or hashing")
    audit_parser.add_argument("--source-root", type=Path, required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create and verify a staging snapshot")
    prepare_parser.add_argument("--source-root", type=Path, required=True)
    prepare_parser.add_argument("--package-root", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify", help="Verify a staged snapshot")
    verify_parser.add_argument("--snapshot-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            result = audit_source(args.source_root)
        elif args.command == "prepare":
            result = prepare_snapshot(args.source_root, args.package_root)
        else:
            result = verify_snapshot(args.snapshot_root)
    except PreparationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
