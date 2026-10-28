"""
=====================================================================
 E10 -- Kiem chung truc tiep giai thich giai phau (song P / QRS)
=====================================================================
Reviewer hoi: "Did you actually measure preservation of P-wave morphology?"
Hien bai di thang tu CHENH LECH HIEU NANG -> GIAI THICH GIAI PHAU ma
khong co bang chung trung gian. Script nay cung cap bang chung do.

Y TUONG: cat cua so quanh dinh R (do tren tin hieu THAT), roi do tuong
quan giua tin hieu that va tin hieu tai tao TRONG TUNG CUA SO:
    cua so P   : R-250ms .. R-80ms   -> hoat dong nhi
    cua so QRS : R-50ms  .. R+60ms   -> dan truyen that
    cua so T   : R+120ms .. R+400ms  -> tai cuc
So sanh tai tao-tu-leadI voi tai tao-tu-leadII.

GIA THUYET cua bai:
    tuong quan cua so P  : lead II  >  lead I
    tuong quan cua so QRS: lead I  >=  lead II
Neu dung -> giai thich giai phau co bang chung truc tiep.

KHONG train lai. Chi can 3 file: ECG that + 2 file tai tao da co.

---------------------------------------------------------------------
 CACH CHAY
---------------------------------------------------------------------
  python paper_v2\\exp_E10_morphology.py ^
      --true  external\\code_test\\ecg_tracings.hdf5 ^
      --gen0  paper_v2\\results_E\\gen_a0.5_lead0.npy ^
      --gen1  paper_v2\\results_E\\gen_a0.5_lead1.npy ^
      --fs    400 ^
      --tag   alpha0.5_seed42 ^
      --out   paper_v2\\results_E\\E10_morphology.csv

Chay lai cho tung seed roi gop lai.
=====================================================================
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
WIN_MS = {"P": (-250, -80), "QRS": (-50, 60), "T": (120, 400)}


def detect_r(sig, fs):
    """Dinh R tren tin hieu that. Uu tien neurokit2, khong co thi dung scipy."""
    try:
        import neurokit2 as nk
        _, info = nk.ecg_peaks(sig, sampling_rate=fs)
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        from scipy.signal import find_peaks
        x = sig - np.median(sig)
        h = 0.5 * np.percentile(np.abs(x), 99)
        pk, _ = find_peaks(x, height=h, distance=int(0.25 * fs))
        return pk


def corr(a, b, eps=1e-12):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > eps else np.nan


def load_gen(path):
    x = np.load(path, allow_pickle=False)
    if x.ndim != 3:
        raise SystemExit(f"{path}: mong doi 3 chieu, nhan {x.shape}")
    if x.shape[1] == 12 and x.shape[2] != 12:
        x = np.transpose(x, (0, 2, 1))
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true", required=True, help="ecg_tracings.hdf5 (goc, chua qua model)")
    ap.add_argument("--gen0", required=True, help=".npy tai tao tu input lead I")
    ap.add_argument("--gen1", required=True, help=".npy tai tao tu input lead II")
    ap.add_argument("--fs", type=int, default=400)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="E10_morphology.csv")
    ap.add_argument("--max_records", type=int, default=0, help="0 = dung het")
    a = ap.parse_args()

    with h5py.File(a.true, "r") as f:
        key = "tracings" if "tracings" in f else list(f.keys())[0]
        Y = np.asarray(f[key])
    G0, G1 = load_gen(a.gen0), load_gen(a.gen1)
    n = min(Y.shape[0], G0.shape[0], G1.shape[0])
    if a.max_records:
        n = min(n, a.max_records)
    Y, G0, G1 = Y[:n], G0[:n], G1[:n]
    print(f"n ban ghi = {n} | shape = {Y.shape} | fs = {a.fs} Hz")

    W = {k: (int(round(v[0] * a.fs / 1000)), int(round(v[1] * a.fs / 1000)))
         for k, v in WIN_MS.items()}
    print("cua so (mau):", W)

    acc = {(w, l, src): [] for w in W for l in range(12) for src in ("leadI", "leadII")}
    nbeat = 0
    for i in range(n):
        r = detect_r(Y[i, :, 1], a.fs)                 # dinh R do tren lead II THAT
        r = r[(r + W["T"][1] < Y.shape[1]) & (r + W["P"][0] >= 0)]
        if len(r) < 2:
            continue
        nbeat += len(r)
        for w, (s0, s1) in W.items():
            for l in range(12):
                c0 = [corr(Y[i, p + s0:p + s1, l], G0[i, p + s0:p + s1, l]) for p in r]
                c1 = [corr(Y[i, p + s0:p + s1, l], G1[i, p + s0:p + s1, l]) for p in r]
                acc[(w, l, "leadI")].append(np.nanmean(c0))
                acc[(w, l, "leadII")].append(np.nanmean(c1))
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{n}")

    rows = []
    for w in W:
        for l in range(12):
            a0 = np.nanmean(acc[(w, l, "leadI")]); a1 = np.nanmean(acc[(w, l, "leadII")])
            rows.append(dict(tag=a.tag, window=w, lead=LEADS[l],
                             corr_from_leadI=a0, corr_from_leadII=a1, delta=a1 - a0))
    df = pd.DataFrame(rows)

    print("\n" + "=" * 84)
    print(f"E10 -- tuong quan hinh thai theo cua so -- {a.tag}  ({nbeat} nhip)")
    print("=" * 84)
    piv = df.pivot_table(index="window", values=["corr_from_leadI", "corr_from_leadII", "delta"])
    print("\nTRUNG BINH TREN CA 12 LEAD:")
    print(piv.round(4).to_string())
    print("\nTHEO TUNG CUA SO, tung lead:")
    for w in ["P", "QRS", "T"]:
        print(f"\n-- cua so {w}")
        print(df[df.window == w].set_index("lead")[
            ["corr_from_leadI", "corr_from_leadII", "delta"]].round(4).to_string())
    print("""
CACH DOC:
  delta > 0 o cua so P   -> tai tao tu lead II giu song P tot hon  (ung ho AF / 1dAVb)
  delta < 0 o cua so QRS -> tai tao tu lead I giu QRS tot hon      (ung ho LBBB / RBBB)
Neu ca hai dung dau thi giai thich giai phau trong bai co bang chung truc tiep.
""")
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    print("Saved:", outp)


if __name__ == "__main__":
    main()
