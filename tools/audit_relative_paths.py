"""Audit repository-relative path contracts for research entrypoints."""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, require_repo_root


VERSIONS = ("v1", "v2", "v3", "v4", "v7", "v8", "v12", "v13")
ENTRYPOINTS = (
    Path("save_genecg.py"),
    Path("paper_v2/check_datasets.py"),
    Path("paper_v2/npy_to_hdf5.py"),
    Path("paper_v2/eval_diagnostic.py"),
    Path("paper_v2/exp_E0_einthoven_check.py"),
    Path("paper_v2/exp_E3_nondipolar.py"),
    Path("tools/audit_relative_paths.py"),
    Path("tools/import_smoke_test.py"),
    Path("tools/train_experiment.py"),
    Path("tools/run_seed_matrix.py"),
    Path("tools/evaluate_model_complexity.py"),
    Path("tools/evaluate_reconstruction.py"),
    Path("tools/select_v15_lambda.py"),
    Path("tools/aggregate_experiments.py"),
    Path("tools/run_code_test_pipeline.py"),
    Path("tools/plot_e3_nondipolar.py"),
    Path("tools/evaluate_multirate.py"),
    Path("tools/evaluate_bland_altman.py"),
    Path("tools/package_artifacts.py"),
    Path("tools/test_new_experiments.py"),
) + tuple(
    Path(version) / area / filename
    for version in VERSIONS
    for area, filename in (
        ("training_code", "train_MCMA_different_kernel_size.py"),
        ("training_code", "train_MCMA_different_window_size.py"),
        ("evaluating_code", "evaluate_signal_MCMA.py"),
        ("evaluating_code", "evaluate_model_HR.py"),
    )
)

PATH_ARGUMENTS = {
    "paths",
    "data_path",
    "modelpath",
    "output_file",
    "trainpath",
    "valpath",
    "testpath",
    "test2path",
    "resultfile",
    "resultfile2",
    "resultpath",
    "pred",
    "gold",
    "out",
    "inp",
    "status",
    "out_dir",
    "metrics_dir",
    "figures_dir",
    "summary_out",
    "combined_out",
    "per_seed_out",
    "wilcoxon_out",
    "data_path",
    "classifier_model",
    "output_root",
    "artifacts_root",
    "manifest",
    "summary",
    "zip_path",
}
OUTPUT_ARGUMENTS = {
    "output_file",
    "resultfile",
    "resultfile2",
    "resultpath",
    "out",
    "out_dir",
    "metrics_dir",
    "figures_dir",
    "summary_out",
    "combined_out",
    "per_seed_out",
    "wilcoxon_out",
    "output_root",
    "manifest",
    "summary",
    "zip_path",
    "artifacts_root",
}
DOCUMENTATION_FILES = (Path("paper_v2/RUNBOOK.md"),) + tuple(
    path for path in (Path("usage.md"), *(Path(version) / "usage.md" for version in VERSIONS))
)

# Task 1 and task 2 explicitly cross the repository boundary to stage a Kaggle
# Dataset and link /kaggle/input into /kaggle/working. They are infrastructure
# boundary tools, not research runtime entrypoints, and are audited separately.
PLATFORM_BOUNDARY_TOOLS = {
    Path("tools/bootstrap_kaggle.py"),
    Path("tools/prepare_kaggle_dataset.py"),
}

WINDOWS_ABSOLUTE_RE = re.compile(
    r"(?<![A-Za-z0-9+.-])(?:[A-Za-z]:[\\/]|\\\\[^\\\s]+[\\/])"
)
PERSONAL_POSIX_RE = re.compile(r"/(?:home|Users|mnt|workspace)/")
BACKSLASH_PATH_RE = re.compile(r"(?:[A-Za-z0-9_.${}-]+\\)+[A-Za-z0-9_.*${}-]*")
MISJOINED_VERSION_RE = re.compile(r"(?:^|/)v\d+/(?:datasets|checkpoints)(?:/|$)")


@dataclass(frozen=True)
class AuditRow:
    check: str
    status: str
    details: str


def _argument_name(call: ast.Call) -> str | None:
    for argument in call.args:
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            return argument.value.lstrip("-").replace("-", "_")
    return None


def _keyword_name(call: ast.Call, keyword_name: str) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == keyword_name and isinstance(keyword.value, ast.Name):
            return keyword.value.id
    return None


def _default_text(call: ast.Call) -> str | None:
    for keyword in call.keywords:
        if keyword.arg != "default":
            continue
        if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return keyword.value.value
        if isinstance(keyword.value, ast.JoinedStr):
            return "".join(
                part.value
                for part in keyword.value.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
    return None


def _path_argument_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        name = _argument_name(node)
        if name in PATH_ARGUMENTS or name == "repo":
            yield node, name


def _research_python_files(root: Path):
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if any(part in {".git", "external", "tests", "artifacts"} for part in relative.parts):
            continue
        if relative in PLATFORM_BOUNDARY_TOOLS or relative == Path("tools/audit_relative_paths.py"):
            continue
        yield relative, path


def _is_relative_forward_path(value: str) -> bool:
    candidate = Path(value)
    windows_candidate = PureWindowsPath(value)
    return bool(value) and not (
        candidate.is_absolute()
        or windows_candidate.is_absolute()
        or windows_candidate.drive
        or "\\" in value
        or ".." in candidate.parts
    )


def run_audit(root: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    missing = [path.as_posix() for path in ENTRYPOINTS if not (root / path).is_file()]
    rows.append(
        AuditRow(
            "entrypoint_inventory",
            "PASS" if not missing else "FAIL",
            f"{len(ENTRYPOINTS)} research entrypoints found"
            if not missing
            else "missing: " + "; ".join(missing),
        )
    )

    root_contract_failures: list[str] = []
    input_contract_failures: list[str] = []
    output_contract_failures: list[str] = []
    default_failures: list[str] = []
    misjoined_defaults: list[str] = []
    missing_input_defaults: list[str] = []
    repo_arguments: list[str] = []
    input_argument_count = 0
    output_argument_count = 0

    for relative in ENTRYPOINTS:
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=relative.as_posix())
        if "require_repo_root()" not in text:
            root_contract_failures.append(relative.as_posix())
        for call, name in _path_argument_calls(tree):
            if name == "repo":
                repo_arguments.append(f"{relative.as_posix()}:{call.lineno}")
                continue
            converter = _keyword_name(call, "type")
            is_training_model = name == "modelpath" and (
                "training_code" in relative.parts
                or relative == Path("tools/train_experiment.py")
            )
            expected = (
                "artifact_path"
                if name in OUTPUT_ARGUMENTS or is_training_model
                else "relative_path"
            )
            if expected == "artifact_path":
                output_argument_count += 1
            else:
                input_argument_count += 1
            if converter != expected:
                failure = f"{relative.as_posix()}:{call.lineno}:{name}:{converter or 'missing'}"
                if expected == "artifact_path":
                    output_contract_failures.append(failure)
                else:
                    input_contract_failures.append(failure)

            default = _default_text(call)
            if default is None:
                continue
            if not _is_relative_forward_path(default):
                default_failures.append(
                    f"{relative.as_posix()}:{call.lineno}:{default}"
                )
            if MISJOINED_VERSION_RE.search(default):
                misjoined_defaults.append(
                    f"{relative.as_posix()}:{call.lineno}:misjoined:{default}"
                )
            if (
                expected == "relative_path"
                and default.startswith(("datasets/", "checkpoints/"))
                and not (root / default).is_file()
            ):
                missing_input_defaults.append(
                    f"{relative.as_posix()}:{call.lineno}:missing:{default}"
                )

    rows.extend(
        (
            AuditRow(
                "repo_root_execution",
                "PASS" if not root_contract_failures and not repo_arguments else "FAIL",
                f"{len(ENTRYPOINTS)} entrypoints require repo root; no --repo override"
                if not root_contract_failures and not repo_arguments
                else "missing guard: " + "; ".join(root_contract_failures)
                + " | --repo args: " + "; ".join(repo_arguments),
            ),
            AuditRow(
                "relative_cli_inputs",
                "PASS" if not input_contract_failures else "FAIL",
                f"all {input_argument_count} research CLI inputs use relative_path"
                if not input_contract_failures
                else "; ".join(input_contract_failures),
            ),
            AuditRow(
                "artifact_cli_outputs",
                "PASS" if not output_contract_failures else "FAIL",
                f"all {output_argument_count} research CLI outputs are confined to artifacts/"
                if not output_contract_failures
                else "; ".join(output_contract_failures),
            ),
            AuditRow(
                "relative_forward_slash_defaults",
                "PASS" if not default_failures else "FAIL",
                "all path defaults are repo-relative, use '/', and avoid parent traversal"
                if not default_failures
                else "; ".join(default_failures),
            ),
            AuditRow(
                "no_vXX_dataset_checkpoint_misjoin",
                "PASS" if not misjoined_defaults else "FAIL",
                "no default starts with vXX/datasets or vXX/checkpoints"
                if not misjoined_defaults
                else "; ".join(misjoined_defaults),
            ),
            AuditRow(
                "default_input_targets_exist",
                "PASS" if not missing_input_defaults else "FAIL",
                "all default dataset/checkpoint inputs resolve from repo root"
                if not missing_input_defaults
                else "; ".join(missing_input_defaults),
            ),
        )
    )

    absolute_failures: list[str] = []
    backslash_failures: list[str] = []
    scanned_python = 0
    for relative, path in _research_python_files(root):
        scanned_python += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if WINDOWS_ABSOLUTE_RE.search(line) or PERSONAL_POSIX_RE.search(line):
                absolute_failures.append(f"{relative.as_posix()}:{line_number}")
            if BACKSLASH_PATH_RE.search(line):
                backslash_failures.append(f"{relative.as_posix()}:{line_number}")

    scanned_docs = 0
    for relative in DOCUMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            continue
        scanned_docs += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if WINDOWS_ABSOLUTE_RE.search(line) or PERSONAL_POSIX_RE.search(line):
                absolute_failures.append(f"{relative.as_posix()}:{line_number}")
            if BACKSLASH_PATH_RE.search(line) and "newcommand" not in line:
                backslash_failures.append(f"{relative.as_posix()}:{line_number}")

    rows.extend(
        (
            AuditRow(
                "no_absolute_runtime_paths",
                "PASS" if not absolute_failures else "FAIL",
                f"scanned {scanned_python} Python files and {scanned_docs} run docs; no personal/absolute runtime path"
                if not absolute_failures
                else "; ".join(absolute_failures),
            ),
            AuditRow(
                "forward_slash_source_and_cli_docs",
                "PASS" if not backslash_failures else "FAIL",
                "runtime path literals and documented CLI paths use '/'"
                if not backslash_failures
                else "; ".join(backslash_failures),
            ),
            AuditRow(
                "platform_boundary_exceptions",
                "PASS",
                "bootstrap_kaggle.py and prepare_kaggle_dataset.py may cross repo boundaries as required by setup tasks 1-2",
            ),
        )
    )
    return rows


def write_report(output: Path, rows: list[AuditRow]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("check", "status", "details"))
        writer.writeheader()
        writer.writerows(
            {"check": row.check, "status": row.status, "details": row.details}
            for row in rows
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=artifact_path,
        default="artifacts/preflight/relative_path_audit.csv",
    )
    return parser.parse_args()


def main() -> int:
    root = require_repo_root()
    args = parse_args()
    rows = run_audit(root)
    write_report(Path(args.output), rows)
    failures = [row for row in rows if row.status != "PASS"]
    for row in rows:
        print(f"{row.status:4}  {row.check}: {row.details}")
    print(f"Report: {Path(args.output).as_posix()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
