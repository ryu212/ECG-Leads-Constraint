"""Evaluate per-record ECG reconstruction, EG identities, and optional base output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, relative_path, require_repo_root
from tools.experiment_common import (
    DERIVED_INDICES,
    DERIVED_NAMES,
    LEAD_NAMES,
    eg_violation_per_record,
    load_model,
    pearson_per_record,
    prediction_and_base,
    write_csv,
    write_evaluation_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--modelpath", type=relative_path, required=True)
    parser.add_argument("--testpath", type=relative_path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--config", default=None,
                        help="Stable configuration label; defaults to --tag.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--input-leads", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=None)
    parser.add_argument("--save-per-record-csv", action="store_true")
    parser.add_argument(
        "--out-dir", type=artifact_path, required=True,
    )
    return parser.parse_args()


def main() -> int:
    require_repo_root()
    args = parse_args()
    if not args.input_leads or any(index not in range(12) for index in args.input_leads):
        raise ValueError("--input-leads must contain indices from 0 to 11")
    model, module, device, config = load_model(
        args.version,
        args.modelpath,
        alpha=args.alpha,
        lambda_=args.lambda_,
    )
    config_label = args.config or args.tag
    loader = module.make_dataloader(
        args.testpath, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    pcc_batches: list[np.ndarray] = []
    mse_batches: list[np.ndarray] = []
    violation_batches: list[np.ndarray] = []
    base_pcc_batches: list[np.ndarray] = []
    record_offset = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            true = module.ecg_to_channels_first(batch).to(device)
            keep = true.std(dim=-1).mean(dim=-1) >= 1e-4
            true = true[keep]
            if true.numel() == 0:
                continue
            batch_pcc = np.empty((true.size(0), len(args.input_leads), 12), dtype=np.float32)
            batch_mse = np.empty_like(batch_pcc)
            batch_violation = np.empty((true.size(0), len(args.input_leads), 4), dtype=np.float32)
            batch_base = np.full_like(batch_pcc, np.nan)
            for column, input_lead in enumerate(args.input_leads):
                indices = torch.full(
                    (true.size(0),), input_lead, dtype=torch.long, device=device
                )
                masked = module.paddingecg(true, indices)
                pred, base = prediction_and_base(model(masked))
                batch_pcc[:, column] = pearson_per_record(pred, true).cpu().numpy()
                batch_mse[:, column] = (pred - true).pow(2).mean(dim=-1).cpu().numpy()
                batch_violation[:, column] = eg_violation_per_record(pred).cpu().numpy()
                if base is not None:
                    batch_base[:, column] = pearson_per_record(base, true).cpu().numpy()
            pcc_batches.append(batch_pcc)
            mse_batches.append(batch_mse)
            violation_batches.append(batch_violation)
            base_pcc_batches.append(batch_base)
            record_offset += true.size(0)
            print(f"batch={batch_index} records={record_offset}")

    if not pcc_batches:
        raise RuntimeError("No valid ECG records were evaluated")
    pcc = np.concatenate(pcc_batches)
    mse = np.concatenate(mse_batches)
    violation = np.concatenate(violation_batches)
    base_pcc = np.concatenate(base_pcc_batches)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / f"{args.tag}_per_record.npz"
    np.savez_compressed(
        npz_path,
        pcc=pcc,
        mse=mse,
        violation_norm=violation,
        base_pcc=base_pcc,
        input_leads=np.asarray(args.input_leads),
        output_leads=np.asarray(LEAD_NAMES),
    )

    rows = []
    for column, input_lead in enumerate(args.input_leads):
        off = [index for index in range(12) if index != input_lead]
        derived_off = [index for index in DERIVED_INDICES if index != input_lead]
        rows.append(
            {
                "tag": args.tag,
                "config": config_label,
                "version": args.version,
                "seed": "" if args.seed is None else args.seed,
                "dataset": args.dataset_name,
                "input_lead": input_lead,
                "input_lead_name": LEAD_NAMES[input_lead],
                "n_records": pcc.shape[0],
                "PCC_all": float(pcc[:, column][:, off].mean()),
                "PCC_derived": float(pcc[:, column][:, derived_off].mean()),
                "MSE_all": float(mse[:, column][:, off].mean()),
                "violation_norm": float(violation[:, column].mean()),
                "base_PCC_all": float(np.nanmean(base_pcc[:, column][:, off])) if not np.isnan(base_pcc[:, column]).all() else "",
            }
        )
    write_csv(args.out_dir / f"{args.tag}_summary.csv", rows)

    identity_rows = []
    for column, input_lead in enumerate(args.input_leads):
        for derived_column, name in enumerate(DERIVED_NAMES):
            values = violation[:, column, derived_column]
            identity_rows.append(
                {
                    "tag": args.tag,
                    "config": config_label,
                    "version": args.version,
                    "seed": "" if args.seed is None else args.seed,
                    "dataset": args.dataset_name,
                    "input_lead": input_lead,
                    "derived_lead": name,
                    "n_records": len(values),
                    "violation_mean": float(values.mean()),
                    "violation_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "violation_max": float(values.max()),
                }
            )
    write_csv(args.out_dir / f"{args.tag}_identity_check.csv", identity_rows)

    if args.save_per_record_csv:
        record_rows = []
        for record_id in range(pcc.shape[0]):
            for column, input_lead in enumerate(args.input_leads):
                off = [index for index in range(12) if index != input_lead]
                record_rows.append(
                    {
                        "tag": args.tag,
                        "config": config_label,
                        "version": args.version,
                        "seed": "" if args.seed is None else args.seed,
                        "dataset": args.dataset_name,
                        "record_id": record_id,
                        "input_lead": input_lead,
                        "PCC_all": float(pcc[record_id, column, off].mean()),
                        "MSE_all": float(mse[record_id, column, off].mean()),
                        "violation_norm": float(violation[record_id, column].mean()),
                    }
                )
        write_csv(args.out_dir / f"{args.tag}_per_record.csv", record_rows)

    write_evaluation_metadata(
        args.out_dir / f"{args.tag}_run_metadata.csv",
        version=args.version,
        modelpath=args.modelpath,
        dataset=args.testpath,
        tag=args.tag,
        extra={
            "seed": "" if args.seed is None else args.seed,
            "dataset_name": args.dataset_name,
            "config": config_label,
            "input_leads": args.input_leads,
            "checkpoint_config": config,
            "records": int(pcc.shape[0]),
        },
    )
    print(f"Saved under: {args.out_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
