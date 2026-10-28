"""Beat-level R/S/T/ST regression-to-mean and Bland-Altman analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from scipy.stats import linregress


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, relative_path, require_repo_root
from tools.experiment_common import LEAD_NAMES, load_model, prediction_and_base, write_evaluation_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--modelpath", type=relative_path, required=True)
    parser.add_argument("--testpath", type=relative_path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--input-leads", type=int, nargs="+", default=[1])
    parser.add_argument("--analysis-leads", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--sampling-rate", type=float, default=500.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=None)
    parser.add_argument(
        "--metrics-dir", type=artifact_path, default=Path("artifacts/metrics/E8")
    )
    parser.add_argument(
        "--figures-dir", type=artifact_path, default=Path("artifacts/figures/E8")
    )
    return parser.parse_args()


def landmark_indices(signal: np.ndarray, fs: float) -> list[dict[str, int]]:
    centered = signal - np.median(signal)
    scale = np.median(np.abs(centered - np.median(centered))) * 1.4826
    prominence = max(float(scale), float(np.std(centered)) * 0.25, 1e-6)
    peaks, _ = find_peaks(
        np.abs(centered), distance=max(1, int(round(0.25 * fs))), prominence=prominence
    )
    points = []
    for peak in peaks:
        s_start = peak + max(1, int(round(0.01 * fs)))
        s_stop = min(len(signal), peak + max(2, int(round(0.08 * fs))))
        t_start = peak + max(1, int(round(0.12 * fs)))
        t_stop = min(len(signal), peak + max(2, int(round(0.40 * fs))))
        st_index = peak + int(round(0.08 * fs))
        if s_start >= s_stop or t_start >= t_stop or st_index >= len(signal):
            continue
        s_index = s_start + int(np.argmin(signal[s_start:s_stop]))
        t_window = signal[t_start:t_stop] - np.median(signal)
        t_index = t_start + int(np.argmax(np.abs(t_window)))
        points.append({"R": int(peak), "S": s_index, "T": t_index, "ST": int(st_index)})
    return points


def summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    regression_rows = []
    bland_rows = []
    groups = ["tag", "version", "seed", "input_lead", "analysis_lead", "wave"]
    for values, part in frame.groupby(groups, dropna=False):
        base = dict(zip(groups, values))
        true = part["true_amplitude"].to_numpy(float)
        pred = part["pred_amplitude"].to_numpy(float)
        error = pred - true
        if len(part) >= 2 and np.std(true) > 0:
            regression = linregress(true, error)
            slope, intercept, r_squared, p_value = (
                regression.slope,
                regression.intercept,
                regression.rvalue**2,
                regression.pvalue,
            )
        else:
            slope = intercept = r_squared = p_value = np.nan
        regression_rows.append(
            {
                **base,
                "n_beats": len(part),
                "error_vs_true_slope": slope,
                "error_vs_true_intercept": intercept,
                "r_squared": r_squared,
                "p_value": p_value,
            }
        )
        bias = float(error.mean())
        sd = float(error.std(ddof=1)) if len(error) > 1 else 0.0
        bland_rows.append(
            {
                **base,
                "n_beats": len(part),
                "bias": bias,
                "sd_difference": sd,
                "loa_lower": bias - 1.96 * sd,
                "loa_upper": bias + 1.96 * sd,
            }
        )
    return pd.DataFrame(regression_rows), pd.DataFrame(bland_rows)


def render_figures(frame: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for (input_lead, analysis_lead), lead_frame in frame.groupby(["input_lead", "analysis_lead"]):
        figure, axes = plt.subplots(4, 2, figsize=(10, 15), constrained_layout=True)
        for row_index, wave in enumerate(("R", "S", "T", "ST")):
            part = lead_frame[lead_frame["wave"] == wave]
            true = part["true_amplitude"].to_numpy(float)
            pred = part["pred_amplitude"].to_numpy(float)
            error = pred - true
            axes[row_index, 0].scatter(true, error, s=8, alpha=0.35)
            if len(true) >= 2 and np.std(true) > 0:
                fit = linregress(true, error)
                xline = np.linspace(true.min(), true.max(), 100)
                axes[row_index, 0].plot(xline, fit.intercept + fit.slope * xline, color="tab:red")
            axes[row_index, 0].axhline(0, color="black", linewidth=0.8)
            axes[row_index, 0].set(title=f"{wave}: error vs true", xlabel="True amplitude", ylabel="Pred - true")
            mean = 0.5 * (pred + true)
            bias = error.mean() if len(error) else np.nan
            sd = error.std(ddof=1) if len(error) > 1 else 0.0
            axes[row_index, 1].scatter(mean, error, s=8, alpha=0.35)
            for value, style in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")):
                axes[row_index, 1].axhline(value, color="tab:red", linestyle=style)
            axes[row_index, 1].set(title=f"{wave}: Bland-Altman", xlabel="Mean amplitude", ylabel="Pred - true")
        stem = f"input_{LEAD_NAMES[input_lead]}_output_{LEAD_NAMES[analysis_lead]}".replace("/", "_")
        for suffix in ("png", "pdf"):
            figure.savefig(output / f"{stem}.{suffix}", dpi=250)
        plt.close(figure)


def main() -> int:
    require_repo_root()
    args = parse_args()
    if any(index not in range(12) for index in [*args.input_leads, *args.analysis_leads]):
        raise ValueError("Lead indices must be between 0 and 11")
    model, module, device, config = load_model(
        args.version, args.modelpath, lambda_=args.lambda_
    )
    loader = module.make_dataloader(
        args.testpath, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    rows = []
    record_offset = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            true = module.ecg_to_channels_first(batch).to(device)
            for input_lead in args.input_leads:
                indices = torch.full((true.size(0),), input_lead, dtype=torch.long, device=device)
                pred, _ = prediction_and_base(model(module.paddingecg(true, indices)))
                true_np = true.detach().cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                for local_index in range(true.size(0)):
                    for analysis_lead in args.analysis_leads:
                        points = landmark_indices(true_np[local_index, analysis_lead], args.sampling_rate)
                        for beat_index, beat in enumerate(points):
                            for wave, sample_index in beat.items():
                                true_value = float(true_np[local_index, analysis_lead, sample_index])
                                pred_value = float(pred_np[local_index, analysis_lead, sample_index])
                                rows.append(
                                    {
                                        "tag": args.tag,
                                        "version": args.version,
                                        "seed": "" if args.seed is None else args.seed,
                                        "record_id": record_offset + local_index,
                                        "beat_index": beat_index,
                                        "sample_index": sample_index,
                                        "input_lead": input_lead,
                                        "input_lead_name": LEAD_NAMES[input_lead],
                                        "analysis_lead": analysis_lead,
                                        "analysis_lead_name": LEAD_NAMES[analysis_lead],
                                        "wave": wave,
                                        "true_amplitude": true_value,
                                        "pred_amplitude": pred_value,
                                        "error": pred_value - true_value,
                                    }
                                )
            record_offset += true.size(0)
            print(f"batch={batch_index} records={record_offset} beats={len(rows)}")
    if not rows:
        raise RuntimeError("No R peaks/beat landmarks were detected")
    frame = pd.DataFrame(rows)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.metrics_dir / "beat_level_metrics.csv", index=False)
    regression, bland = summaries(frame)
    regression.to_csv(args.metrics_dir / "regression_summary.csv", index=False)
    bland.to_csv(args.metrics_dir / "bland_altman_summary.csv", index=False)
    render_figures(frame, args.figures_dir)
    write_evaluation_metadata(
        args.metrics_dir / f"{args.tag}_run_metadata.csv",
        version=args.version,
        modelpath=args.modelpath,
        dataset=args.testpath,
        tag=args.tag,
        extra={
            "sampling_rate": args.sampling_rate,
            "input_leads": args.input_leads,
            "analysis_leads": args.analysis_leads,
            "landmark_definition": "R=abs-prominent peak; S=true minimum 10-80ms; T=true max-absolute 120-400ms; ST=R+80ms",
            "checkpoint_config": config,
        },
    )
    print(f"Saved metrics under: {args.metrics_dir.as_posix()}")
    print(f"Saved figures under: {args.figures_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

