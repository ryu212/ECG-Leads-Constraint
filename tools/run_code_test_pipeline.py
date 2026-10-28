"""Generate CODE-test ECG, convert HDF5, classify, score, and write checksums."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, relative_path, require_repo_root
from tools.experiment_common import sha256, utc_now, write_csv


CLASSES = ("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST")
CLASS_THRESHOLDS = np.array(
    [0.124, 0.070, 0.050, 0.278, 0.390, 0.174],
    dtype=np.float32,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=["E2", "E4", "E5", "E6", "E7"])
    parser.add_argument("--version", required=True)
    parser.add_argument("--modelpath", type=relative_path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lead", type=int, required=True, choices=range(12))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--data-path", type=relative_path, default=Path("external/code_test/ecg_tracings.hdf5"))
    parser.add_argument("--gold", type=relative_path, default=Path("external/code_test/gold_standard.csv"))
    parser.add_argument("--classifier-model", type=relative_path, default=Path("external/model.hdf5"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expected-records", type=int, default=827)
    parser.add_argument("--segment-len", type=int, default=4096)
    parser.add_argument("--real-baseline", action="store_true")
    parser.add_argument(
        "--restore-input-lead",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore the measured input lead: mode B (default); disable for mode A.",
    )
    parser.add_argument("--output-root", type=artifact_path, default=Path("artifacts"))
    return parser.parse_args()


def run_logged(command: list[str], *, cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(chr(10) + f"[{utc_now()}] {' '.join(command)}" + chr(10))
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"Command failed with code {result.returncode}; see {log}")


def npy_to_hdf5(inp: Path, out: Path, expected_records: int, segment_len: int) -> tuple[int, ...]:
    values = np.load(inp, mmap_mode="r", allow_pickle=False)
    if values.shape != (expected_records, segment_len, 12):
        raise ValueError(
            f"Generated ECG shape {values.shape}; expected {(expected_records, segment_len, 12)}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as handle:
        dataset = handle.create_dataset(
            "tracings",
            shape=values.shape,
            dtype="float32",
            chunks=(1, min(segment_len, 1024), 12),
            compression="gzip",
            compression_opts=1,
        )
        for start in range(0, len(values), 32):
            dataset[start : start + 32] = values[start : start + 32]
        handle.flush()
    with h5py.File(out, "r") as handle:
        dataset = handle["tracings"]
        if tuple(dataset.shape) != tuple(values.shape):
            raise RuntimeError(f"HDF5 verification shape mismatch: {dataset.shape}")
        if not np.isfinite(dataset[0]).all() or not np.isfinite(dataset[-1]).all():
            raise FloatingPointError("HDF5 verification found NaN/Inf at boundary records")
    return tuple(values.shape)


def load_gold(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    values = frame[list(CLASSES)].to_numpy() if set(CLASSES).issubset(frame.columns) else frame.to_numpy()
    values = np.asarray(values, dtype=int)
    if values.ndim != 2 or values.shape[1] != len(CLASSES):
        raise ValueError(f"Gold standard must have six labels, got {values.shape}")
    return values


def score_predictions(prediction_path: Path, gold_path: Path, tag: str) -> tuple[list[dict], dict]:
    probability = np.load(prediction_path, allow_pickle=False)
    if probability.ndim == 3 and probability.shape[0] == 1:
        probability = probability[0]
    gold = load_gold(gold_path)
    if probability.shape != gold.shape:
        raise ValueError(f"Prediction shape {probability.shape} != gold shape {gold.shape}")
    rows = []
    for index, class_name in enumerate(CLASSES):
        threshold = float(CLASS_THRESHOLDS[index])
        predicted = probability[:, index] > threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold[:, index], predicted, average="binary", zero_division=0
        )
        rows.append(
            {
                "tag": tag,
                "class": class_name,
                "n_records": len(gold),
                "n_positive": int(gold[:, index].sum()),
                "threshold": threshold,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auprc": float(average_precision_score(gold[:, index], probability[:, index])),
            }
        )
    macro = {
        "tag": tag,
        "n_records": len(gold),
        "precision_macro": float(np.mean([row["precision"] for row in rows])),
        "recall_macro": float(np.mean([row["recall"] for row in rows])),
        "f1_macro": float(np.mean([row["f1"] for row in rows])),
        "auprc_macro": float(np.mean([row["auprc"] for row in rows])),
    }
    return rows, macro


def upsert_csv(path: Path, new_rows: list[dict], key: str = "tag") -> None:
    existing: list[dict] = []
    if path.is_file():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
    keys = {str(row[key]) for row in new_rows}
    existing = [row for row in existing if str(row.get(key)) not in keys]
    write_csv(path, [*existing, *new_rows], fieldnames=new_rows[0].keys())


def classifier_command(hdf5_path: Path, model_path: Path, output_path: Path, batch_size: int) -> list[str]:
    return [
        sys.executable,
        "predict.py",
        str(hdf5_path.resolve()),
        str(model_path.resolve()),
        "--dataset_name",
        "tracings",
        "--output_file",
        str(output_path.resolve()),
        "-bs",
        str(batch_size),
    ]


def main() -> int:
    require_repo_root()
    args = parse_args()
    tag = args.tag or f"{args.version}_{args.config}_seed{args.seed}_lead{args.lead}"
    evaluation_mode = "B" if args.restore_input_lead else "A"
    tag = f"{tag}_mode{evaluation_mode}"
    generated_dir = args.output_root / "generated_ecg" / args.experiment
    prediction_dir = args.output_root / "predictions" / args.experiment
    metric_dir = args.output_root / "metrics" / args.experiment / "diagnostic"
    log = args.output_root / "logs" / args.experiment / f"{tag}.log"
    for directory in (generated_dir, prediction_dir, metric_dir):
        directory.mkdir(parents=True, exist_ok=True)

    npy_path = generated_dir / f"{tag}.npy"
    hdf5_path = generated_dir / f"{tag}.hdf5"
    prediction_path = prediction_dir / f"{tag}.npy"
    generate_command = [
        sys.executable,
        "save_genecg.py",
        "--version", args.version,
        "--data_path", args.data_path.as_posix(),
        "--modelpath", args.modelpath.as_posix(),
        "--output_file", npy_path.as_posix(),
        "--lead", str(args.lead),
        "--batch_size", str(args.batch_size),
        "--ecglen", "1024",
        "--segment_len", str(args.segment_len),
        "--resampling",
        "--restore-input-lead" if args.restore_input_lead else "--no-restore-input-lead",
    ]
    run_logged(generate_command, cwd=REPOSITORY_DIR, log=log)
    shape = npy_to_hdf5(npy_path, hdf5_path, args.expected_records, args.segment_len)

    classifier_root = REPOSITORY_DIR / "external" / "automatic-ecg-diagnosis"
    run_logged(
        classifier_command(hdf5_path, args.classifier_model, prediction_path, args.batch_size),
        cwd=classifier_root,
        log=log,
    )
    per_class, macro = score_predictions(prediction_path, args.gold, tag)
    shared = {
        "experiment": args.experiment,
        "version": args.version,
        "config": args.config,
        "seed": args.seed,
        "lead": args.lead,
        "evaluation_mode": evaluation_mode,
        "restore_input_lead": args.restore_input_lead,
    }
    per_class = [{**shared, **row} for row in per_class]
    macro_row = {**shared, **macro}
    upsert_csv(metric_dir / "per_class_scores.csv", per_class)
    upsert_csv(metric_dir / "macro_scores.csv", [macro_row])

    if args.real_baseline:
        real_prediction = prediction_dir / "real_12lead.npy"
        run_logged(
            classifier_command(args.data_path, args.classifier_model, real_prediction, args.batch_size),
            cwd=classifier_root,
            log=log,
        )
        _, real_macro = score_predictions(real_prediction, args.gold, "real_12lead")
        write_csv(metric_dir / "real_baseline.csv", [real_macro])

    manifest_row = {
        **shared,
        "created_at_utc": utc_now(),
        "tag": tag,
        "shape": "x".join(map(str, shape)),
        "records": shape[0],
        "npy_path": npy_path.as_posix(),
        "npy_sha256": sha256(npy_path),
        "hdf5_path": hdf5_path.as_posix(),
        "hdf5_sha256": sha256(hdf5_path),
        "prediction_path": prediction_path.as_posix(),
        "prediction_sha256": sha256(prediction_path),
        "checkpoint": args.modelpath.as_posix(),
        "checkpoint_sha256": sha256(args.modelpath),
        "gold_sha256": sha256(args.gold),
        "classifier_sha256": sha256(args.classifier_model),
        "resampling": "400->500->400",
    }
    upsert_csv(generated_dir / "generation_manifest.csv", [manifest_row])
    print(f"Saved generated ECG: {npy_path.as_posix()} and {hdf5_path.as_posix()}")
    print(f"Saved predictions: {prediction_path.as_posix()}")
    print(f"Saved metrics: {metric_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
