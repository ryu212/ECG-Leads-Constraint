"""Aggregate experiment CSVs and optionally run predefined paired Wilcoxon tests."""

from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, require_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="CSV paths or glob patterns")
    parser.add_argument("--group-by", nargs="+", required=True)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--summary-out", type=artifact_path, required=True)
    parser.add_argument("--combined-out", type=artifact_path, default=None)
    parser.add_argument("--seed-column", default="seed")
    parser.add_argument("--per-seed-out", type=artifact_path, default=None)
    parser.add_argument("--pair-column", default=None)
    parser.add_argument("--pair", nargs=2, action="append", default=[])
    parser.add_argument(
        "--record-keys", nargs="+", default=["dataset", "record_id", "input_lead"]
    )
    parser.add_argument("--wilcoxon-out", type=artifact_path, default=None)
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        paths.update(Path(value) for value in matches if Path(value).is_file())
    return sorted(paths)


def aggregate(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    grouped = frame.groupby(groups, dropna=False, sort=True)
    for group_values, part in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(groups, group_values))
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce").dropna().to_numpy(float)
            count = len(values)
            mean = float(values.mean()) if count else math.nan
            std = float(values.std(ddof=1)) if count > 1 else 0.0 if count == 1 else math.nan
            row[f"{metric}_n"] = count
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - 1.96 * std / math.sqrt(count) if count else math.nan
            row[f"{metric}_ci95_high"] = mean + 1.96 * std / math.sqrt(count) if count else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    total = positive + negative
    return float((positive - negative) / total) if total else 0.0


def paired_tests(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if not args.pair_column or not args.pair:
        return pd.DataFrame()
    missing = set([args.pair_column, *args.record_keys, *args.metrics]) - set(frame.columns)
    if missing:
        raise ValueError(f"Wilcoxon input is missing columns: {sorted(missing)}")
    rows = []
    context_groups = [column for column in args.group_by if column != args.pair_column]
    contexts = frame.groupby(context_groups, dropna=False) if context_groups else [((), frame)]
    for context_values, context in contexts:
        if not isinstance(context_values, tuple):
            context_values = (context_values,)
        context_row = dict(zip(context_groups, context_values))
        for left_name, right_name in args.pair:
            for metric in args.metrics:
                left = context[context[args.pair_column].astype(str) == left_name]
                right = context[context[args.pair_column].astype(str) == right_name]
                merged = left[[*args.record_keys, metric]].merge(
                    right[[*args.record_keys, metric]],
                    on=args.record_keys,
                    suffixes=("_left", "_right"),
                    validate="one_to_one",
                ).dropna()
                if merged.empty:
                    continue
                left_values = pd.to_numeric(merged[f"{metric}_left"]).to_numpy(float)
                right_values = pd.to_numeric(merged[f"{metric}_right"]).to_numpy(float)
                differences = right_values - left_values
                if np.allclose(differences, 0):
                    statistic, pvalue = 0.0, 1.0
                else:
                    statistic, pvalue = wilcoxon(left_values, right_values, zero_method="wilcox")
                rows.append(
                    {
                        **context_row,
                        "metric": metric,
                        "left_config": left_name,
                        "right_config": right_name,
                        "n_pairs": len(merged),
                        "mean_difference_right_minus_left": float(differences.mean()),
                        "median_difference_right_minus_left": float(np.median(differences)),
                        "rank_biserial_effect": rank_biserial(differences),
                        "wilcoxon_statistic": float(statistic),
                        "p_value": float(pvalue),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    require_repo_root()
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError("No CSV files matched --inputs")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = path.as_posix()
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    missing = set([*args.group_by, *args.metrics]) - set(combined.columns)
    if missing:
        raise ValueError(f"Aggregate input is missing columns: {sorted(missing)}")

    if args.combined_out:
        args.combined_out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.combined_out, index=False)
    summary = aggregate(combined, args.group_by, args.metrics)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False)

    if args.per_seed_out:
        if args.seed_column not in combined.columns:
            raise ValueError(f"Missing seed column: {args.seed_column}")
        seed_groups = list(dict.fromkeys([*args.group_by, args.seed_column]))
        per_seed = aggregate(combined, seed_groups, args.metrics)
        args.per_seed_out.parent.mkdir(parents=True, exist_ok=True)
        per_seed.to_csv(args.per_seed_out, index=False)

    tests = paired_tests(combined, args)
    if args.wilcoxon_out:
        args.wilcoxon_out.parent.mkdir(parents=True, exist_ok=True)
        tests.to_csv(args.wilcoxon_out, index=False)
    print(f"Inputs: {len(paths)} | rows: {len(combined)}")
    print(f"Saved: {args.summary_out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

