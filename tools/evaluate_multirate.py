"""Evaluate reconstruction robustness after controlled sampling-rate degradation."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, relative_path, require_repo_root
from tools.experiment_common import (
    LEAD_NAMES,
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--source-rate", type=int, default=500)
    parser.add_argument("--rates", type=int, nargs="+", default=[250, 300, 400, 500])
    parser.add_argument("--input-leads", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=None)
    parser.add_argument(
        "--out-dir", type=artifact_path, default=Path("artifacts/metrics/E7")
    )
    return parser.parse_args()


def roundtrip_rate(batch: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if target_rate == source_rate:
        return batch
    values = batch.detach().cpu().numpy()
    divisor = math.gcd(source_rate, target_rate)
    degraded = resample_poly(
        values, target_rate // divisor, source_rate // divisor, axis=1
    )
    restored = resample_poly(
        degraded, source_rate // divisor, target_rate // divisor, axis=1
    )
    original_length = batch.shape[1]
    if restored.shape[1] > original_length:
        restored = restored[:, :original_length]
    elif restored.shape[1] < original_length:
        restored = np.pad(restored, ((0, 0), (0, original_length - restored.shape[1]), (0, 0)))
    return torch.from_numpy(np.asarray(restored)).to(dtype=batch.dtype)


def main() -> int:
    require_repo_root()
    args = parse_args()
    if any(rate <= 0 or rate > args.source_rate for rate in args.rates):
        raise ValueError("Evaluation rates must be positive and no higher than --source-rate")
    model, module, device, config = load_model(
        args.version, args.modelpath, lambda_=args.lambda_
    )
    started_at = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loader = module.make_dataloader(
        args.testpath, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    rows = []
    record_offset = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            keep = batch.float().std(dim=1).mean(dim=1) >= 1e-4
            batch = batch[keep].float()
            if batch.numel() == 0:
                continue
            for rate in args.rates:
                degraded = roundtrip_rate(batch, args.source_rate, rate)
                alias_rmse = (degraded - batch).pow(2).mean(dim=(1, 2)).sqrt().numpy()
                true = module.ecg_to_channels_first(batch).to(device)
                degraded_cf = module.ecg_to_channels_first(degraded).to(device)
                for input_lead in args.input_leads:
                    indices = torch.full(
                        (true.size(0),), input_lead, dtype=torch.long, device=device
                    )
                    pred, _ = prediction_and_base(model(module.paddingecg(degraded_cf, indices)))
                    pcc = pearson_per_record(pred, true).cpu().numpy()
                    mse = (pred - true).pow(2).mean(dim=-1).cpu().numpy()
                    off = [index for index in range(12) if index != input_lead]
                    for local_index in range(true.size(0)):
                        rows.append(
                            {
                                "tag": args.tag,
                                "version": args.version,
                                "seed": "" if args.seed is None else args.seed,
                                "record_id": record_offset + local_index,
                                "source_rate_hz": args.source_rate,
                                "evaluation_rate_hz": rate,
                                "input_lead": input_lead,
                                "input_lead_name": LEAD_NAMES[input_lead],
                                "PCC_all": float(pcc[local_index, off].mean()),
                                "MSE_all": float(mse[local_index, off].mean()),
                                "input_roundtrip_rmse": float(alias_rmse[local_index]),
                            }
                        )
            record_offset += len(batch)
            print(f"batch={batch_index} records={record_offset}")
    if not rows:
        raise RuntimeError("No valid records were evaluated")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_record = pd.DataFrame(rows)
    per_record.to_csv(args.out_dir / f"{args.tag}_per_record.csv", index=False)
    summary = (
        per_record.groupby(
            ["tag", "version", "seed", "source_rate_hz", "evaluation_rate_hz", "input_lead", "input_lead_name"],
            dropna=False,
            as_index=False,
        )[["PCC_all", "MSE_all", "input_roundtrip_rmse"]]
        .agg(["mean", "std", "count"])
    )
    summary.columns = ["_".join(map(str, value)).rstrip("_") for value in summary.columns]
    per_run_path = args.out_dir / "runs" / f"{args.tag}_per_rate_metrics.csv"
    per_run_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(per_run_path, index=False)
    elapsed_seconds = time.perf_counter() - started_at
    cuda_peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    write_evaluation_metadata(
        args.out_dir / f"{args.tag}_run_metadata.csv",
        version=args.version,
        modelpath=args.modelpath,
        dataset=args.testpath,
        tag=args.tag,
        extra={
            "seed": "" if args.seed is None else args.seed,
            "rates": args.rates,
            "source_rate": args.source_rate,
            "records": record_offset,
            "elapsed_seconds": elapsed_seconds,
            "cuda_peak_memory_bytes": cuda_peak_memory_bytes,
            "checkpoint_config": config,
        },
    )
    print(f"Saved: {per_run_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
