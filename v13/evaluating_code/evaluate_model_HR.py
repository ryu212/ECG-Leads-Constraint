"""
PyTorch version of evaluate_model_HR.py

Evaluates heart-rate consistency between real and generated ECG signals.
Uses `utils_MCMA` dataloaders and helper functions.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import neurokit2 as nk

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, relative_path, require_repo_root

VERSION = Path(__file__).resolve().parents[1].name

from v13.utils_MCMA import (
    Trainer,
    ecg_to_channels_first,
    make_dataloader,
    paddingecg,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testpath", type=relative_path, required=True)
    parser.add_argument("--modelpath", type=relative_path, required=True)
    parser.add_argument(
        "--resultpath",
        type=artifact_path,
        default=f"artifacts/metrics/{VERSION}/hr",
    )
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--ecglen", type=int, default=1024)
    parser.add_argument("--fs", type=int, default=500, help="sampling rate for HR extraction")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def extracthr(ecg, fs=500):
    try:
        signals, info = nk.ecg_peaks(ecg, sampling_rate=fs)
        r_peaks = info["ECG_R_Peaks"]
        if len(r_peaks) < 2:
            return np.nan
        rr_intervals = np.diff(r_peaks) / float(fs)
        heart_rate = 60.0 / np.mean(rr_intervals)
        return heart_rate
    except Exception:
        return np.nan


def get_hrmetric(hr1):
    rhr = np.zeros((4, 1))
    rhr[0] = np.sum(np.std(hr1, axis=1))
    rhr[1] = np.sum(np.std(hr1, axis=1) / np.mean(hr1, axis=1))
    rhr[2] = np.sum(np.max(hr1, axis=1) - np.min(hr1, axis=1))
    rhr[3] = hr1.shape[0]
    return rhr


def final_get(a):
    return a[:-1, :] / a[-1, :][None, :]


def prediction_tensor(model_output):
    """Return the generated ECG for models yielding a tensor or tuple/list."""
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


def write2excel_hr2(rhr, fhr, maehr, excel_path):
    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    df1 = pd.DataFrame(rhr).round(4)
    df2 = pd.DataFrame(fhr).round(4)
    df3 = pd.DataFrame(maehr).round(4)
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df1.to_excel(writer, sheet_name="real_heartrate", index=True)
        df2.to_excel(writer, sheet_name="fake_heartrate", index=True)
        df3.to_excel(writer, sheet_name="MAE_HR", index=True)


def test_ae_hr(model, ds, ecglen=1024, numlead=12, fs=500, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()

    rhr = np.zeros((4, numlead))
    gen_fhr = np.zeros((4, numlead))
    maehr = np.zeros((1, numlead))

    with torch.no_grad():
        for step, ecg12 in enumerate(ds):
            print("The current step is", step)
            # ecg12: (batch, length, 12)
            batch_orig = ecg12.shape[0]
            seq_len = ecg12.shape[1]
            padding_len = (ecglen - seq_len % ecglen) % ecglen
            if padding_len > 0:
                pad = torch.zeros((ecg12.shape[0], padding_len, ecg12.shape[2]), dtype=ecg12.dtype)
                ecg12_padded = torch.cat([ecg12, pad], dim=1)
            else:
                ecg12_padded = ecg12

            # reshape to segments: (batch * nseg, ecglen, 12)
            nseg = ecg12_padded.shape[1] // ecglen
            ecg12_new = ecg12_padded.reshape(-1, ecglen, 12)

            # convert to channels-first for paddingecg/model
            ecg12_ch = ecg_to_channels_first(ecg12_new)  # (Nseg, 12, ecglen)

            for i in range(numlead):
                lead_index = torch.full((ecg12_ch.size(0),), i, dtype=torch.long)
                ecg1 = paddingecg(ecg12_ch, lead_index).to(device)

                gen_ecg12_ch = prediction_tensor(model(ecg1))
                gen_ecg12_time = gen_ecg12_ch.permute(0, 2, 1).cpu().numpy()  # (Nseg, ecglen, 12)

                # reshape back to (batch, seq_len + padding_len, 12)
                gen_ecg12_full = gen_ecg12_time.reshape(batch_orig, nseg * ecglen, 12)
                if padding_len > 0:
                    gen_ecg12_full = gen_ecg12_full[:, : -padding_len, :]

                # compute HR per original sample and lead
                hr1 = np.zeros((batch_orig, 12))
                hr2 = np.zeros((batch_orig, 12))
                for i2 in range(batch_orig):
                    for j in range(12):
                        real_sig = ecg12[i2, :, j].cpu().numpy()
                        fake_sig = gen_ecg12_full[i2, :, j]
                        hr1[i2, j] = extracthr(real_sig, fs=fs)
                        hr2[i2, j] = extracthr(fake_sig, fs=fs)

                idx = np.where(np.isnan(hr1 + hr2) == 1)[0]
                if idx.size > 0:
                    hr1 = np.delete(hr1, idx, axis=0)
                    hr2 = np.delete(hr2, idx, axis=0)
                if hr1.size == 0:
                    continue
                maehr[0, i] += np.sum(np.abs(hr1 - hr2))
                rhr[:, i : i + 1] += get_hrmetric(hr1)
                gen_fhr[:, i : i + 1] += get_hrmetric(hr2)

    maehr = maehr / (12 * rhr[-1, :])
    rhr = final_get(rhr)
    gen_fhr = final_get(gen_fhr)
    return maehr, rhr, gen_fhr


if __name__ == "__main__":
    require_repo_root()
    args = parse_args()
    device = args.device

    test_loader = make_dataloader(args.testpath, batch_size=args.bs, shuffle=False)

    # build model using Trainer to get same architecture
    # Trainer requires a modelpath; we'll instantiate with same ecglen/kernel/pool defaults
    trainer = Trainer(modelpath=args.modelpath, ecglen=args.ecglen)
    # load weights
    try:
        model = trainer.load(args.modelpath)
    except Exception:
        # if the file is a full model saved differently, try torch.load direct
        ck = torch.load(args.modelpath, map_location=torch.device(device or "cpu"))
        # if it's state_dict dict, instantiate model then load
        if isinstance(ck, dict) and "model_state_dict" in ck:
            model = trainer.model
            model.load_state_dict(ck["model_state_dict"])
        else:
            model = ck

    maehr, rhr, fhr = test_ae_hr(model=model, ds=test_loader, ecglen=args.ecglen, numlead=12, fs=args.fs, device=device)
    excel_path = Path(args.resultpath) / "evaluated_hr.xlsx"
    write2excel_hr2(rhr, fhr, maehr, excel_path)
    print("Saved:", excel_path)

