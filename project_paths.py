"""Repository-relative path policy shared by command-line entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path, PureWindowsPath


REPO_MARKER = Path("paper_v2/RUNBOOK_CHECKLIST.csv")
ARTIFACT_ROOT = Path("artifacts")


def require_repo_root(cwd: Path | None = None) -> Path:
    """Return the active repo root or fail with a clear invocation error."""
    root = Path.cwd() if cwd is None else Path(cwd)
    if not (root / REPO_MARKER).is_file():
        raise RuntimeError(
            "Run this command from the WRFB_MCMA repository root "
            f"(missing marker: {REPO_MARKER.as_posix()})."
        )
    return root.resolve()


def relative_path(value: str | Path) -> Path:
    """Argparse converter for safe, slash-normalised repo-relative paths."""
    text = str(value)
    candidate = Path(text)
    windows_candidate = PureWindowsPath(text)
    if (
        not text
        or candidate.is_absolute()
        or windows_candidate.is_absolute()
        or bool(windows_candidate.drive)
    ):
        raise argparse.ArgumentTypeError(f"Path must be relative to repo root: {text!r}")
    if "\\" in text:
        raise argparse.ArgumentTypeError(f"Use '/' in repository paths: {text!r}")
    if ".." in candidate.parts:
        raise argparse.ArgumentTypeError(f"Path must not escape repo root: {text!r}")
    return candidate


def artifact_path(value: str | Path) -> Path:
    """Argparse converter that confines outputs to the artifacts tree."""
    candidate = relative_path(value)
    if not candidate.parts or candidate.parts[0] != ARTIFACT_ROOT.name:
        raise argparse.ArgumentTypeError(
            f"Output must be under {ARTIFACT_ROOT.as_posix()}/: {str(value)!r}"
        )
    return candidate
