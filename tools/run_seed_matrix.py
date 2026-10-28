"""Run selected training jobs from ``seed_run_status.csv`` sequentially.

The default mode is a dry run. Pass ``--execute`` to start GPU training.
Blocked parity jobs require the additional ``--allow-blocked-parity`` flag.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import require_repo_root


DEFAULT_STATUS = Path("artifacts/manifests/seed_run_status.csv")
DEFAULT_DRY_RUN_LOG = Path("artifacts/logs/seed_matrix_dry_run.log")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise argparse.ArgumentTypeError("paths must be relative to the repository root")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--status", type=relative_path, default=DEFAULT_STATUS)
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--version", action="append", default=[])
    parser.add_argument("--exclude-version", action="append", default=[])
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute jobs. Without this flag, only print and log the selection.",
    )
    parser.add_argument(
        "--allow-blocked-parity",
        action="store_true",
        help="Allow TRAIN rows marked BLOCKED_PARITY after parity was reviewed externally.",
    )
    return parser.parse_args()


def read_status(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"Status CSV has no header: {path}")
        return list(reader), list(reader.fieldnames)


def write_status(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_matches(value: str, requested: list[str]) -> bool:
    if not requested:
        return True
    available = {item.strip() for item in value.split(";") if item.strip()}
    return bool(available.intersection(requested))


def completed_and_valid(row: dict[str, str]) -> bool:
    required = [row["modelpath"], row["last_checkpoint"], row["metadata_path"]]
    paths = [REPOSITORY_DIR / value for value in required if value]
    if len(paths) != 3 or not all(path.is_file() for path in paths):
        return False
    expected_best = row.get("best_sha256", "").strip()
    expected_last = row.get("last_sha256", "").strip()
    if expected_best and sha256(paths[0]) != expected_best:
        return False
    if expected_last and sha256(paths[1]) != expected_last:
        return False
    return row.get("run_status") == "SUCCESS"


def selected_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: int(item["run_order"])):
        if row.get("seed") != str(args.seed) or row.get("action") != "TRAIN":
            continue
        if args.version and row.get("version") not in args.version:
            continue
        if row.get("version") in args.exclude_version:
            continue
        if not experiment_matches(row.get("experiment", ""), args.experiment):
            continue
        readiness = row.get("readiness", "")
        allowed = readiness in {"READY", "READY_TO_TRAIN"}
        allowed = allowed or (readiness == "BLOCKED_PARITY" and args.allow_blocked_parity)
        if not allowed:
            print(
                f"SKIP {row['job_id']}: readiness={readiness}; "
                "prerequisite or source is not ready"
            )
            continue
        if completed_and_valid(row):
            print(f"SKIP {row['job_id']}: completed checkpoint and metadata are valid")
            continue
        selected.append(row)
    if args.max_jobs is not None:
        if args.max_jobs <= 0:
            raise ValueError("--max-jobs must be positive")
        selected = selected[: args.max_jobs]
    return selected


def command_for(row: dict[str, str]) -> list[str]:
    command = shlex.split(row["launch_script"], posix=True)
    if not command or command[0] not in {"python", "python3"}:
        raise RuntimeError(f"Unsafe or unsupported launch command for {row['job_id']}")
    command[0] = sys.executable
    last_checkpoint = REPOSITORY_DIR / row["last_checkpoint"]
    if last_checkpoint.is_file() and "--resume" not in command:
        command.extend(["--resume", row["last_checkpoint"]])
        row["resume_from"] = row["last_checkpoint"]
    return command


def render_command(command: list[str]) -> str:
    return shlex.join(command)


def write_dry_run(rows: list[dict[str, str]], commands: list[list[str]], seed: int) -> Path:
    output = REPOSITORY_DIR / DEFAULT_DRY_RUN_LOG
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"generated_at_utc={utc_now()}", f"seed={seed}", f"jobs={len(rows)}"]
    newline = chr(10)
    lines.extend(
        f"{index}. {row['job_id']}{newline}   {render_command(command)}"
        for index, (row, command) in enumerate(zip(rows, commands), start=1)
    )
    output.write_text(newline.join(lines) + newline, encoding="utf-8")
    return output


def execute_job(
    row: dict[str, str],
    command: list[str],
    status_path: Path,
    rows: list[dict[str, str]],
    fields: list[str],
) -> None:
    log_path = REPOSITORY_DIR / row["log_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    (REPOSITORY_DIR / row["modelpath"]).parent.mkdir(parents=True, exist_ok=True)

    row["selected"] = "TRUE"
    row["run_status"] = "RUNNING"
    row["started_at_utc"] = utc_now()
    row["finished_at_utc"] = ""
    row["return_code"] = ""
    row["error"] = ""
    row["attempt"] = str(int(row.get("attempt") or 0) + 1)
    write_status(status_path, rows, fields)

    print(f"RUN  {row['job_id']}")
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(chr(10) + f"[{utc_now()}] {render_command(command)}" + chr(10))
        result = subprocess.run(
            command,
            cwd=REPOSITORY_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    row["return_code"] = str(result.returncode)
    row["finished_at_utc"] = utc_now()
    best = REPOSITORY_DIR / row["modelpath"]
    last = REPOSITORY_DIR / row["last_checkpoint"]
    metadata = REPOSITORY_DIR / row["metadata_path"]
    outputs_ok = all(path.is_file() for path in (best, last, metadata))
    if result.returncode == 0 and outputs_ok:
        row["run_status"] = "SUCCESS"
        row["best_sha256"] = sha256(best)
        row["last_sha256"] = sha256(last)
        print(f"PASS {row['job_id']}")
    else:
        row["run_status"] = "FAILED"
        row["error"] = (
            f"return_code={result.returncode}; outputs_ok={outputs_ok}; log={row['log_path']}"
        )
        write_status(status_path, rows, fields)
        raise RuntimeError(f"Job failed: {row['job_id']}; see {row['log_path']}")
    write_status(status_path, rows, fields)


def main() -> int:
    require_repo_root()
    args = parse_args()
    status_path = REPOSITORY_DIR / args.status
    rows, fields = read_status(status_path)
    chosen = selected_rows(rows, args)
    commands = [command_for(row) for row in chosen]
    dry_run_log = write_dry_run(chosen, commands, args.seed)

    print(f"Selected {len(chosen)} training job(s) for seed {args.seed}")
    for row, command in zip(chosen, commands):
        print(f"  {row['job_id']}: {render_command(command)}")
    print(f"Dry-run log: {dry_run_log.relative_to(REPOSITORY_DIR)}")

    if not args.execute:
        print("DRY RUN only; add --execute to start training")
        return 0
    if not chosen:
        raise RuntimeError("No runnable jobs selected")

    for row, command in zip(chosen, commands):
        execute_job(row, command, status_path, rows, fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
