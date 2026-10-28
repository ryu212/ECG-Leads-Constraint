"""Create a writable Kaggle working copy backed by relative input symlinks.

The module is designed to be loaded in the Kaggle notebook kernel with
``runpy.run_path`` and then called through :func:`bootstrap`.  It never writes
to ``/kaggle/input`` and never copies the large dataset/checkpoint assets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROJECT_ROOT_NAME = "WRFB_MCMA"
MARKER_RELATIVE_PATH = Path("paper_v2/RUNBOOK_CHECKLIST.csv")
MANIFEST_RELATIVE_PATH = Path("artifacts/manifests/source_manifest.csv")
SOURCE_CONFIGURATION_FILES: frozenset[Path] = frozenset(
    {
        MANIFEST_RELATIVE_PATH,
        *(
            Path("artifacts/manifests") / name
            for name in (
                "architecture_decision_v14.csv",
                "seed_training_plan.csv",
                "seed_run_status.csv",
                "e6_ablation_plan.csv",
                "experiment_tool_map.csv",
            )
        ),
    }
)
REPORT_RELATIVE_PATH = Path("artifacts/preflight/bootstrap_report.csv")
DEFAULT_INPUT_BASE = Path("/kaggle/input")
DEFAULT_WORKING_ROOT = Path("/kaggle/working/WRFB_MCMA")
MAX_SOURCE_FILE_BYTES = 20 * 1024 * 1024

ARTIFACT_RELATIVE_PATHS: tuple[Path, ...] = tuple(
    Path("artifacts") / name
    for name in (
        "checkpoints",
        "metrics",
        "generated_ecg",
        "predictions",
        "figures",
        "logs",
        "manifests",
        "preflight",
        "smoke",
    )
)

REQUIRED_ASSET_FILES: tuple[Path, ...] = tuple(
    Path(path)
    for path in (
        "paper_v2/RUNBOOK_CHECKLIST.csv",
        "tools/bootstrap_kaggle.py",
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
    )
)

LINK_SPECS: tuple[tuple[Path, Path], ...] = tuple(
    (Path(working), Path(source))
    for working, source in (
        ("datasets", "datasets"),
        ("checkpoints", "checkpoints"),
        ("external/code_test", "external/code_test"),
        ("external/model.hdf5", "external/model.hdf5"),
    )
)

EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "tmp",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
EXCLUDED_SOURCE_SUFFIXES = {
    ".pt",
    ".pth",
    ".hdf5",
    ".zip",
    ".npy",
    ".pyc",
    ".pyo",
    ".tmp",
    ".temp",
    ".log",
    ".bak",
    ".swp",
}
EXCLUDED_SOURCE_FILE_NAMES = {".DS_Store", "Thumbs.db"}
EXCLUDED_EXTERNAL_PREFIXES = (
    Path("external/automatic-ecg-diagnosis/outputs"),
    Path("external/automatic-ecg-diagnosis/dnn_predicts"),
)


class BootstrapError(RuntimeError):
    """Raised when bootstrapping cannot safely continue."""


@dataclass(frozen=True)
class ManifestRecord:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ReportRow:
    timestamp_utc: str
    input_root: str
    working_root: str
    record_type: str
    path: str
    source_path: str
    working_path: str
    link_target: str
    size_bytes: int | str
    status: str
    detail: str


class Reporter:
    def __init__(self, working_root: Path) -> None:
        self.working_root = working_root
        self.input_root: Path | None = None
        self.rows: list[ReportRow] = []

    def set_input_root(self, input_root: Path) -> None:
        self.input_root = input_root

    def add(
        self,
        record_type: str,
        *,
        path: str = "",
        source_path: Path | str = "",
        working_path: Path | str = "",
        link_target: str = "",
        size_bytes: int | str = "",
        status: str,
        detail: str = "",
    ) -> None:
        self.rows.append(
            ReportRow(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                input_root=str(self.input_root or ""),
                working_root=str(self.working_root),
                record_type=record_type,
                path=path,
                source_path=str(source_path),
                working_path=str(working_path),
                link_target=link_target,
                size_bytes=size_bytes,
                status=status,
                detail=detail,
            )
        )

    def write(self) -> Path:
        report_path = self.working_root / REPORT_RELATIVE_PATH
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(ReportRow.__annotations__))
            writer.writeheader()
            writer.writerows(asdict(row) for row in self.rows)
        return report_path


def _normalise_relative(path: Path) -> str:
    value = path.as_posix()
    if path.is_absolute() or value == ".." or value.startswith("../") or "\\" in value:
        raise BootstrapError(f"Unsafe relative path: {value}")
    return value


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def discover_input_root(input_base: Path) -> tuple[Path, list[Path]]:
    input_base = input_base.resolve()
    if not input_base.is_dir():
        raise BootstrapError(f"Kaggle input directory does not exist: {input_base}")
    markers = sorted(input_base.glob(f"**/{MARKER_RELATIVE_PATH.as_posix()}"))
    roots = sorted({marker.parents[1].resolve() for marker in markers})
    if len(roots) != 1:
        raise BootstrapError(
            f"Expected exactly one repository marker, found {len(roots)}; "
            f"candidates={[str(root) for root in roots]}"
        )
    input_root = roots[0]
    if input_root.name != PROJECT_ROOT_NAME:
        raise BootstrapError(
            f"Input repository must be named {PROJECT_ROOT_NAME!r}, got {input_root.name!r}"
        )
    if not _is_relative_to(input_root, input_base):
        raise BootstrapError(f"Discovered repository escapes input directory: {input_root}")
    return input_root, markers


def _read_manifest(input_root: Path) -> dict[str, ManifestRecord]:
    manifest_path = input_root / MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        raise BootstrapError(f"Source manifest is missing: {manifest_path}")
    records: dict[str, ManifestRecord] = {}
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "size_bytes", "sha256"]:
            raise BootstrapError(f"Unexpected source manifest schema: {reader.fieldnames}")
        for row in reader:
            relative = row["path"]
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or "\\" in relative:
                raise BootstrapError(f"Unsafe manifest path: {relative}")
            if relative in records:
                raise BootstrapError(f"Duplicate manifest path: {relative}")
            try:
                size_bytes = int(row["size_bytes"])
            except ValueError as exc:
                raise BootstrapError(f"Invalid size in manifest for {relative}") from exc
            sha256 = row["sha256"].lower()
            if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
                raise BootstrapError(f"Invalid SHA256 in manifest for {relative}")
            records[relative] = ManifestRecord(relative, size_bytes, sha256)
    return records


def _validate_assets(
    input_root: Path,
    manifest: dict[str, ManifestRecord],
    reporter: Reporter,
    verify_sha256: bool,
) -> None:
    failures: list[str] = []
    for relative in REQUIRED_ASSET_FILES:
        relative_text = _normalise_relative(relative)
        source = input_root / relative
        record = manifest.get(relative_text)
        status, detail, size_bytes = "PASS", "size_matches_manifest", ""
        if record is None:
            status, detail = "FAIL", "missing_from_manifest"
        elif not source.is_file():
            status, detail = "FAIL", "missing_or_not_a_file"
        else:
            try:
                size_bytes = source.stat().st_size
                if size_bytes != record.size_bytes:
                    status, detail = "FAIL", "size_mismatch"
                elif verify_sha256:
                    if _sha256(source) != record.sha256:
                        status, detail = "FAIL", "sha256_mismatch"
                    else:
                        detail = "sha256_matches_manifest"
            except OSError as exc:
                status, detail = "FAIL", f"unreadable:{exc}"
        reporter.add(
            "required_asset",
            path=relative_text,
            source_path=source,
            size_bytes=size_bytes,
            status=status,
            detail=detail,
        )
        if status != "PASS":
            failures.append(f"{relative_text} ({detail})")
    if failures:
        raise BootstrapError("Required asset validation failed: " + "; ".join(failures))


def _under_prefix(relative: Path, prefix: Path) -> bool:
    return relative == prefix or prefix in relative.parents


def _source_exclusion_reason(relative: Path) -> str | None:
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
        return "cache_or_environment"
    if any(_under_prefix(relative, working) for working, _ in LINK_SPECS):
        return "linked_asset"
    if relative.parts and relative.parts[0] == "artifacts":
        if relative not in SOURCE_CONFIGURATION_FILES:
            return "input_artifact"
    if (
        len(relative.parts) >= 2
        and re.fullmatch(r"v\d+", relative.parts[0])
        and relative.parts[1] == "results"
    ):
        return "legacy_result"
    if any(_under_prefix(relative, prefix) for prefix in EXCLUDED_EXTERNAL_PREFIXES):
        return "external_generated_output"
    if relative.name.startswith("~$"):
        return "office_lock"
    if relative.name in EXCLUDED_SOURCE_FILE_NAMES:
        return "platform_cache"
    if relative.suffix.lower() in EXCLUDED_SOURCE_SUFFIXES:
        return "binary_or_temporary_payload"
    return None


def _collect_small_source(input_root: Path) -> tuple[list[Path], int]:
    sources: list[Path] = []
    skipped = 0
    for source in input_root.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(input_root)
        if _source_exclusion_reason(relative):
            skipped += 1
            continue
        size_bytes = source.stat().st_size
        if size_bytes > MAX_SOURCE_FILE_BYTES:
            raise BootstrapError(
                f"Unclassified file exceeds the source-copy limit "
                f"({MAX_SOURCE_FILE_BYTES} bytes): {relative} ({size_bytes} bytes)"
            )
        sources.append(source)
    return sorted(sources), skipped


def _copy_small_source(
    input_root: Path, working_root: Path, reporter: Reporter
) -> tuple[int, int]:
    sources, skipped = _collect_small_source(input_root)
    copied = 0
    for source in sources:
        relative = source.relative_to(input_root)
        destination = working_root / relative
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise BootstrapError(f"Source destination has incompatible type: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if source.stat().st_size != destination.stat().st_size or _sha256(source) != _sha256(
                destination
            ):
                raise BootstrapError(
                    f"Refusing to overwrite modified working source: {destination}"
                )
            detail = "already_current"
        else:
            shutil.copy2(source, destination)
            copied += 1
            detail = "copied"
        reporter.add(
            "source_copy",
            path=_normalise_relative(relative),
            source_path=source,
            working_path=destination,
            size_bytes=source.stat().st_size,
            status="PASS",
            detail=detail,
        )
    return copied, skipped


def _create_relative_links(
    input_root: Path, working_root: Path, reporter: Reporter
) -> dict[str, str]:
    created: dict[str, str] = {}
    for working_relative, input_relative in LINK_SPECS:
        source = input_root / input_relative
        destination = working_root / working_relative
        if not source.exists():
            raise BootstrapError(f"Symlink source is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        relative_target = os.path.relpath(source, destination.parent)
        if os.path.isabs(relative_target):
            raise BootstrapError(f"Could not construct relative symlink for {destination}")

        if destination.is_symlink():
            existing_target = os.readlink(destination)
            if os.path.isabs(existing_target) or destination.resolve() != source.resolve():
                raise BootstrapError(
                    f"Existing symlink is incorrect: {destination} -> {existing_target}"
                )
            detail = "already_current"
        elif destination.exists():
            raise BootstrapError(
                f"Refusing to replace non-symlink asset destination: {destination}"
            )
        else:
            destination.symlink_to(relative_target, target_is_directory=source.is_dir())
            detail = "created"

        if os.path.isabs(os.readlink(destination)) or destination.resolve() != source.resolve():
            raise BootstrapError(f"Relative symlink verification failed: {destination}")
        working_text = _normalise_relative(working_relative)
        created[working_text] = relative_target
        reporter.add(
            "symlink",
            path=working_text,
            source_path=source,
            working_path=destination,
            link_target=relative_target,
            status="PASS",
            detail=detail,
        )
    return created


def _linked_via(relative: Path) -> tuple[Path, str]:
    for working_relative, input_relative in LINK_SPECS:
        if _under_prefix(relative, input_relative):
            suffix = relative.relative_to(input_relative)
            return working_relative / suffix, _normalise_relative(working_relative)
    return relative, ""


def _validate_working_assets(
    working_root: Path,
    input_root: Path,
    reporter: Reporter,
    links: dict[str, str],
) -> None:
    failures: list[str] = []
    for relative in REQUIRED_ASSET_FILES:
        working_relative, link_name = _linked_via(relative)
        working_path = working_root / working_relative
        source_path = input_root / relative
        if not working_path.is_file():
            status, detail = "FAIL", "not_accessible_from_working_copy"
            failures.append(_normalise_relative(relative))
        else:
            status, detail = "PASS", "accessible_via_working_copy"
        reporter.add(
            "working_asset",
            path=_normalise_relative(relative),
            source_path=source_path,
            working_path=working_path,
            link_target=links.get(link_name, ""),
            size_bytes=working_path.stat().st_size if working_path.is_file() else "",
            status=status,
            detail=detail,
        )
    if failures:
        raise BootstrapError(
            "Assets are not accessible from working copy: " + "; ".join(failures)
        )


def _create_writable_artifact_directories(
    working_root: Path, reporter: Reporter
) -> int:
    """Create the canonical output tree and prove every directory is writable."""
    artifact_root = working_root / "artifacts"
    if artifact_root.is_symlink():
        raise BootstrapError(f"Artifact root must not be a symlink: {artifact_root}")

    artifact_root.mkdir(parents=True, exist_ok=True)
    resolved_working_root = working_root.resolve()
    created_count = 0

    for relative in ARTIFACT_RELATIVE_PATHS:
        directory = working_root / relative
        if directory.is_symlink():
            raise BootstrapError(f"Artifact directory must not be a symlink: {directory}")
        existed = directory.is_dir()
        directory.mkdir(parents=True, exist_ok=True)
        if not _is_relative_to(directory.resolve(), resolved_working_root):
            raise BootstrapError(f"Artifact directory escapes working repo: {directory}")

        probe_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=".write-test-",
                dir=directory,
                delete=False,
            ) as probe:
                probe.write("ok\n")
                probe_path = Path(probe.name)
        except OSError as exc:
            raise BootstrapError(
                f"Artifact directory is not writable: {directory}: {exc}"
            ) from exc
        finally:
            if probe_path is not None:
                try:
                    probe_path.unlink()
                except FileNotFoundError:
                    pass

        if not existed:
            created_count += 1
        reporter.add(
            "artifact_directory",
            path=_normalise_relative(relative),
            working_path=directory,
            status="PASS",
            detail="writable_existing" if existed else "writable_created",
        )

    return created_count


def bootstrap(
    input_base: Path = DEFAULT_INPUT_BASE,
    working_root: Path = DEFAULT_WORKING_ROOT,
    verify_sha256: bool = False,
) -> dict[str, object]:
    """Build and activate a writable working copy; raise on unsafe state."""
    input_base = Path(input_base).resolve()
    working_root = Path(working_root).resolve()
    reporter = Reporter(working_root)
    try:
        input_root, markers = discover_input_root(input_base)
        reporter.set_input_root(input_root)
        reporter.add(
            "repo_discovery",
            path=_normalise_relative(MARKER_RELATIVE_PATH),
            source_path=markers[0],
            status="PASS",
            detail="exactly_one_repository",
        )
        if _is_relative_to(working_root, input_root) or _is_relative_to(input_root, working_root):
            raise BootstrapError(
                f"Input and working roots must not contain one another: "
                f"input={input_root}, working={working_root}"
            )

        manifest = _read_manifest(input_root)
        reporter.add(
            "manifest",
            path=_normalise_relative(MANIFEST_RELATIVE_PATH),
            source_path=input_root / MANIFEST_RELATIVE_PATH,
            size_bytes=(input_root / MANIFEST_RELATIVE_PATH).stat().st_size,
            status="PASS",
            detail=f"loaded_{len(manifest)}_records",
        )
        _validate_assets(input_root, manifest, reporter, verify_sha256)

        working_root.mkdir(parents=True, exist_ok=True)
        copied_count, skipped_count = _copy_small_source(input_root, working_root, reporter)
        artifact_directory_created_count = _create_writable_artifact_directories(
            working_root, reporter
        )
        links = _create_relative_links(input_root, working_root, reporter)
        _validate_working_assets(working_root, input_root, reporter, links)

        os.chdir(working_root)
        working_text = str(working_root)
        while sys.path.count(working_text) > 1:
            sys.path.remove(working_text)
        if working_text not in sys.path:
            sys.path.insert(0, working_text)
        if Path.cwd() != working_root or sys.path.count(working_text) != 1:
            raise BootstrapError("Failed to activate working repository")
        reporter.add(
            "working_directory",
            working_path=working_root,
            status="PASS",
            detail="cwd_and_sys_path_active",
        )
        report_path = reporter.write()
        return {
            "input_root": str(input_root),
            "working_root": str(working_root),
            "report_path": str(report_path),
            "copied_source_file_count": copied_count,
            "skipped_input_file_count": skipped_count,
            "artifact_directory_count": len(ARTIFACT_RELATIVE_PATHS),
            "artifact_directory_created_count": artifact_directory_created_count,
            "relative_symlink_count": len(links),
            "required_asset_count": len(REQUIRED_ASSET_FILES),
            "status": "PASS",
        }
    except Exception as exc:
        reporter.add("bootstrap", status="FAIL", detail=str(exc))
        try:
            reporter.write()
        except OSError:
            pass
        if isinstance(exc, BootstrapError):
            raise
        raise BootstrapError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-base", type=Path, default=DEFAULT_INPUT_BASE)
    parser.add_argument("--working-root", type=Path, default=DEFAULT_WORKING_ROOT)
    parser.add_argument("--verify-sha256", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = bootstrap(args.input_base, args.working_root, args.verify_sha256)
    except BootstrapError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
