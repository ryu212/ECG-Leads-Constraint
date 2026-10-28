"""Smoke-test Linux-safe package imports for the currently supported versions."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, require_repo_root


# v14 is intentionally deferred until its implementation exists.
VERSIONS = ("v1", "v3", "v4", "v7", "v8", "v13")
SUBPACKAGES = ("loss", "training_code", "evaluating_code")
ENTRYPOINT_FILES = (
    "training_code/train_MCMA_different_kernel_size.py",
    "training_code/train_MCMA_different_window_size.py",
    "evaluating_code/evaluate_signal_MCMA.py",
    "evaluating_code/evaluate_model_HR.py",
)
MODULE_SUFFIXES = (
    "",
    ".layers",
    ".model",
    ".loss.loss",
    ".utils_MCMA",
    ".training_code.train_MCMA_different_kernel_size",
    ".training_code.train_MCMA_different_window_size",
    ".evaluating_code.evaluate_signal_MCMA",
    ".evaluating_code.evaluate_model_HR",
)
FORBIDDEN_TOP_LEVEL_IMPORTS = {"MCMA", "utils_MCMA", "model", "layers", "loss"}
REPORT_COLUMNS = (
    "version",
    "package_init",
    "qualified_imports",
    "repo_root_once",
    "case_sensitive_names",
    "runtime_import",
    "status",
    "details",
)


@dataclass(frozen=True)
class SmokeRow:
    version: str
    package_init: str
    qualified_imports: str
    repo_root_once: str
    case_sensitive_names: str
    runtime_import: str
    status: str
    details: str


def _exact_child(parent: Path, name: str) -> Path | None:
    """Return a child only when its spelling matches exactly."""
    if not parent.is_dir():
        return None
    matches = [child for child in parent.iterdir() if child.name == name]
    return matches[0] if len(matches) == 1 else None


def _exact_relative_path(root: Path, relative: Path) -> Path | None:
    current = root
    for part in relative.parts:
        current = _exact_child(current, part)
        if current is None:
            return None
    return current


def _module_exists_with_exact_case(root: Path, module: str) -> bool:
    parts = module.split(".")
    current = root
    for part in parts[:-1]:
        current = _exact_child(current, part)
        if current is None or not current.is_dir():
            return False

    final = parts[-1]
    module_file = _exact_child(current, f"{final}.py")
    package_dir = _exact_child(current, final)
    return bool(
        module_file is not None
        or (
            package_dir is not None
            and package_dir.is_dir()
            and _exact_child(package_dir, "__init__.py") is not None
        )
    )


def _source_module(relative: Path) -> tuple[str, ...]:
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return tuple(parts)


def _resolved_import_module(
    source_relative: Path, node: ast.ImportFrom
) -> str | None:
    if node.level == 0:
        return node.module
    package = list(_source_module(source_relative)[:-1])
    if source_relative.name == "__init__.py":
        package = list(_source_module(source_relative))
    parent_levels = node.level - 1
    if parent_levels > len(package):
        return None
    if parent_levels:
        package = package[:-parent_levels]
    if node.module:
        package.extend(node.module.split("."))
    return ".".join(package)


def _python_files(root: Path, version: str):
    yield from sorted((root / version).rglob("*.py"))


def _package_marker_failures(root: Path, version: str) -> list[str]:
    expected = [Path(version) / "__init__.py"] + [
        Path(version) / subpackage / "__init__.py" for subpackage in SUBPACKAGES
    ]
    return [
        path.as_posix()
        for path in expected
        if _exact_relative_path(root, path) is None
    ]


def _qualified_import_failures(root: Path, version: str) -> list[str]:
    failures: list[str] = []
    for path in _python_files(root, version):
        relative = path.relative_to(root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative.as_posix())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level = alias.name.split(".", 1)[0]
                    if top_level in FORBIDDEN_TOP_LEVEL_IMPORTS:
                        failures.append(
                            f"{relative.as_posix()}:{node.lineno}:{alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                imported = _resolved_import_module(relative, node)
                if imported is None:
                    failures.append(
                        f"{relative.as_posix()}:{node.lineno}:invalid-relative-import"
                    )
                    continue
                top_level = imported.split(".", 1)[0] if imported else ""
                if node.level == 0 and top_level in FORBIDDEN_TOP_LEVEL_IMPORTS:
                    failures.append(
                        f"{relative.as_posix()}:{node.lineno}:{imported}"
                    )
    return failures


def _case_failures(root: Path, version: str) -> list[str]:
    failures: list[str] = []
    expected_modules = [f"{version}{suffix}" for suffix in MODULE_SUFFIXES]
    for module in expected_modules:
        if not _module_exists_with_exact_case(root, module):
            failures.append(module)

    for path in _python_files(root, version):
        relative = path.relative_to(root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative.as_posix())
        for node in ast.walk(tree):
            imported_modules: list[str] = []
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported = _resolved_import_module(relative, node)
                if imported:
                    imported_modules.append(imported)
            for module in imported_modules:
                if module.split(".", 1)[0] not in VERSIONS:
                    continue
                if not _module_exists_with_exact_case(root, module):
                    failures.append(f"{relative.as_posix()}:{node.lineno}:{module}")
    return failures


def _repo_root_failures(root: Path, version: str) -> list[str]:
    failures: list[str] = []
    for entrypoint in ENTRYPOINT_FILES:
        relative = Path(version) / entrypoint
        path = root / relative
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=relative.as_posix())
        insert_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "insert"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "path"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "sys"
        ]
        if len(insert_calls) != 1 or "Path(__file__).resolve().parents[2]" not in text:
            failures.append(f"{relative.as_posix()}:insert_count={len(insert_calls)}")
    return failures


def _ensure_runtime_root_once(root: Path) -> int:
    root_resolved = root.resolve()
    retained: list[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry or Path.cwd()).resolve()
        except (OSError, RuntimeError):
            retained.append(entry)
            continue
        if resolved != root_resolved:
            retained.append(entry)
    sys.path[:] = [str(root_resolved), *retained]
    return sum(
        1
        for entry in sys.path
        if Path(entry or Path.cwd()).resolve() == root_resolved
    )


def _runtime_import(version: str) -> tuple[str, str]:
    failures: list[str] = []
    for suffix in MODULE_SUFFIXES:
        module = f"{version}{suffix}"
        try:
            importlib.import_module(module)
        except Exception as exc:  # report the exact environment/import failure
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
    return ("PASS", "") if not failures else ("FAIL", " | ".join(failures))


def run_smoke(root: Path, static_only: bool = False) -> list[SmokeRow]:
    runtime_root_count = _ensure_runtime_root_once(root)
    rows: list[SmokeRow] = []
    for version in VERSIONS:
        marker_failures = _package_marker_failures(root, version)
        import_failures = _qualified_import_failures(root, version)
        root_failures = _repo_root_failures(root, version)
        case_failures = _case_failures(root, version)

        package_init = "PASS" if not marker_failures else "FAIL"
        qualified_imports = "PASS" if not import_failures else "FAIL"
        repo_root_once = (
            "PASS" if not root_failures and runtime_root_count == 1 else "FAIL"
        )
        case_sensitive_names = "PASS" if not case_failures else "FAIL"

        details = marker_failures + import_failures + root_failures + case_failures
        if static_only:
            runtime_import = "NOT_RUN"
        else:
            runtime_import, runtime_detail = _runtime_import(version)
            if runtime_detail:
                details.append(runtime_detail)

        static_pass = all(
            value == "PASS"
            for value in (
                package_init,
                qualified_imports,
                repo_root_once,
                case_sensitive_names,
            )
        )
        if not static_pass or runtime_import == "FAIL":
            status = "FAIL"
        elif runtime_import == "NOT_RUN":
            status = "PENDING"
        else:
            status = "PASS"

        rows.append(
            SmokeRow(
                version=version,
                package_init=package_init,
                qualified_imports=qualified_imports,
                repo_root_once=repo_root_once,
                case_sensitive_names=case_sensitive_names,
                runtime_import=runtime_import,
                status=status,
                details="; ".join(details) if details else "all checks passed",
            )
        )
    return rows


def write_report(output: Path, rows: list[SmokeRow]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=artifact_path,
        default="artifacts/preflight/import_smoke_test.csv",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Run source/package checks without importing dependency-heavy modules.",
    )
    return parser.parse_args()


def main() -> int:
    root = require_repo_root()
    args = parse_args()
    rows = run_smoke(root, static_only=args.static_only)
    write_report(Path(args.output), rows)
    for row in rows:
        print(
            f"{row.status:7} {row.version}: static="
            f"{row.package_init}/{row.qualified_imports}/"
            f"{row.repo_root_once}/{row.case_sensitive_names}; "
            f"runtime={row.runtime_import}"
        )
        if row.details != "all checks passed":
            print(f"         {row.details}")
    print(f"Report: {Path(args.output).as_posix()}")
    return 1 if any(row.status == "FAIL" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
