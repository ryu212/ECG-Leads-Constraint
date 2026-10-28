"""Summarize validation-only v15 lambda runs without launching any training."""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, require_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-glob", required=True)
    parser.add_argument("--reconstruction-glob", default=None)
    parser.add_argument("--classifier-glob", default=None)
    parser.add_argument(
        "--out", type=artifact_path, default=Path("artifacts/metrics/E1a/lambda_selection.csv")
    )
    return parser.parse_args()


def main() -> int:
    require_repo_root()
    args = parse_args()
    rows = []
    by_lambda = {}
    for value in sorted(glob.glob(args.metadata_glob, recursive=True)):
        path = Path(value)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle), None)
        if not row or row.get("version") != "v15":
            continue
        if row.get("lambda", "") == "" or row.get("best_val_loss", "") == "":
            continue
        rows.append(
            {
                "lambda": float(row["lambda"]),
                "seed": row.get("seed", ""),
                "best_val_loss": float(row["best_val_loss"]),
                "last_val_pcc": float(row["last_val_pcc"]) if row.get("last_val_pcc") else "",
                "PCC_all": "",
                "PCC_derived": "",
                "f1_macro": "",
                "auprc_macro": "",
                "selection_metric": "best_val_loss",
                "metadata_path": path.as_posix(),
                "selected": False,
            }
        )
        by_lambda[rows[-1]["lambda"]] = rows[-1]
    if not rows:
        raise RuntimeError("No completed v15 metadata rows matched --metadata-glob")

    if args.reconstruction_glob:
        pcc = {}
        counts = {}
        for value in sorted(glob.glob(args.reconstruction_glob, recursive=True)):
            path = Path(value)
            match = re.search(r"lambda_([0-9.]+)", path.as_posix())
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    lambda_value = row.get("lambda") or (match.group(1) if match else "")
                    if not lambda_value:
                        continue
                    lambda_value = float(lambda_value)
                    if lambda_value not in by_lambda:
                        continue
                    pcc.setdefault(lambda_value, [0.0, 0.0])
                    counts[lambda_value] = counts.get(lambda_value, 0) + 1
                    pcc[lambda_value][0] += float(row.get("PCC_all") or 0.0)
                    pcc[lambda_value][1] += float(row.get("PCC_derived") or row.get("PCC_all") or 0.0)
        for lambda_value, values in pcc.items():
            by_lambda[lambda_value]["PCC_all"] = values[0] / counts[lambda_value]
            by_lambda[lambda_value]["PCC_derived"] = values[1] / counts[lambda_value]
            by_lambda[lambda_value]["selection_metric"] = "PCC"

    if args.classifier_glob:
        scores = {}
        counts = {}
        for value in sorted(glob.glob(args.classifier_glob, recursive=True)):
            path = Path(value)
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    match = re.search(r"lambda_([0-9.]+)", " ".join(str(row.get(key, "")) for key in ("tag", "config")))
                    if not match:
                        match = re.search(r"lambda_([0-9.]+)", path.as_posix())
                    if not match:
                        continue
                    lambda_value = float(match.group(1))
                    if lambda_value not in by_lambda:
                        continue
                    scores.setdefault(lambda_value, [0.0, 0.0])
                    counts[lambda_value] = counts.get(lambda_value, 0) + 1
                    scores[lambda_value][0] += float(row.get("f1_macro") or 0.0)
                    scores[lambda_value][1] += float(row.get("auprc_macro") or 0.0)
        for lambda_value, values in scores.items():
            by_lambda[lambda_value]["f1_macro"] = values[0] / counts[lambda_value]
            by_lambda[lambda_value]["auprc_macro"] = values[1] / counts[lambda_value]
            by_lambda[lambda_value]["selection_metric"] = "classifier_f1_macro"

    winner = max(
        range(len(rows)),
        key=lambda index: (
            rows[index]["f1_macro"] != "",
            rows[index]["f1_macro"] if rows[index]["f1_macro"] != "" else float("-inf"),
            rows[index]["auprc_macro"] if rows[index]["auprc_macro"] != "" else float("-inf"),
            rows[index]["PCC_derived"] != "",
            rows[index]["PCC_derived"] if rows[index]["PCC_derived"] != "" else float("-inf"),
            rows[index]["PCC_all"] if rows[index]["PCC_all"] != "" else float("-inf"),
            rows[index]["last_val_pcc"] if rows[index]["last_val_pcc"] != "" else float("-inf"),
            -rows[index]["best_val_loss"],
        ),
    )
    rows[winner]["selected"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Selected lambda={rows[winner]['lambda']} by {rows[winner]['selection_metric']}")
    print(f"Saved: {args.out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
