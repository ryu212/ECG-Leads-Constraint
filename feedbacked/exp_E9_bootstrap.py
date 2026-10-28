"""
=====================================================================
 E9 -- Bootstrap theo BENH NHAN cho chenh lech giua hai input lead
=====================================================================
Reviewer chi ra: p-value hien tai tinh tren 6 lan chay (training runs),
KHONG phai tren benh nhan. Script nay tinh khoang tin cay o muc benh nhan.

CACH LAM: lay mau lai 827 ban ghi CO HOAN LAI (paired bootstrap -- cung
bo chi so benh nhan ap cho ca lead I va lead II), moi lan tinh lai F1 va
AUPRC, roi lay percentile 2.5 / 97.5 cua HIEU (lead II - lead I).

KHONG can chay lai model. Chi can cac file .npy xac suat du doan ma
predict.py da sinh ra.

---------------------------------------------------------------------
 CACH CHAY
---------------------------------------------------------------------
  python paper_v2\\exp_E9_bootstrap.py ^
      --lead0 pred_alpha0.5_seed42_lead0.npy pred_alpha0.5_seed2026_lead0.npy pred_alpha0.5_seed1_lead0.npy ^
      --lead1 pred_alpha0.5_seed42_lead1.npy pred_alpha0.5_seed2026_lead1.npy pred_alpha0.5_seed1_lead1.npy ^
      --gold  external\\code_test\\gold_standard.csv ^
      --tag   alpha0.5_modeB ^
      --out   paper_v2\\results_E\\E9_bootstrap.csv

Thu tu file trong --lead0 va --lead1 PHAI khop nhau theo seed.
=====================================================================
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]
# nguong co dinh cua Ribeiro -- giong het pipeline dang dung
THR = np.array([0.124, 0.070, 0.050, 0.278, 0.390, 0.174])


def f1_at(y, p, thr, idx):
    """F1 cho tung lop tren tap chi so idx. y,p: (N,6)."""
    yy, pp = y[idx], (p[idx] >= thr)
    tp = (yy & pp).sum(0).astype(float)
    fp = (~yy & pp).sum(0).astype(float)
    fn = (yy & ~pp).sum(0).astype(float)
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    return np.divide(2 * prec * rec, prec + rec,
                     out=np.zeros_like(tp), where=(prec + rec) > 0)


def auprc_at(y, p, idx):
    """AUPRC (average precision) cho tung lop, tu cai dat de khoi phu thuoc sklearn."""
    out = np.zeros(y.shape[1])
    for j in range(y.shape[1]):
        yy, pp = y[idx, j], p[idx, j]
        if yy.sum() == 0:
            out[j] = np.nan; continue
        o = np.argsort(-pp)
        yy = yy[o]
        tp = np.cumsum(yy)
        prec = tp / np.arange(1, len(yy) + 1)
        out[j] = (prec * yy).sum() / yy.sum()
    return out


def load(paths):
    arrs = []
    for f in paths:
        a = np.load(f)
        if a.ndim == 3:
            a = a[0]
        arrs.append(a)
    return arrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead0", nargs="+", required=True, help=".npy cho input lead I, theo tung seed")
    ap.add_argument("--lead1", nargs="+", required=True, help=".npy cho input lead II, CUNG thu tu seed")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="E9_bootstrap.csv")
    ap.add_argument("--B", type=int, default=2000, help="so lan bootstrap")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if len(a.lead0) != len(a.lead1):
        raise SystemExit("So file cua --lead0 va --lead1 phai bang nhau (khop theo seed).")

    P0, P1 = load(a.lead0), load(a.lead1)
    g = pd.read_csv(a.gold)
    y = g[CLASSES].values if all(c in g.columns for c in CLASSES) else g.values
    y = np.asarray(y).astype(bool)
    N = y.shape[0]
    for p in P0 + P1:
        if p.shape != y.shape:
            raise SystemExit(f"Kich thuoc lech: pred {p.shape} vs gold {y.shape}")

    print(f"n benh nhan = {N} | so seed = {len(P0)} | B = {a.B}")
    print("so ca duong tinh:", dict(zip(CLASSES, y.sum(0))))

    rng = np.random.default_rng(a.seed)
    dF1 = np.zeros((a.B, 6)); dAU = np.zeros((a.B, 6))
    for b in range(a.B):
        idx = rng.integers(0, N, N)                 # cung chi so cho ca hai lead
        f1_0 = np.mean([f1_at(y, p, THR, idx) for p in P0], axis=0)
        f1_1 = np.mean([f1_at(y, p, THR, idx) for p in P1], axis=0)
        au_0 = np.nanmean([auprc_at(y, p, idx) for p in P0], axis=0)
        au_1 = np.nanmean([auprc_at(y, p, idx) for p in P1], axis=0)
        dF1[b] = f1_1 - f1_0
        dAU[b] = au_1 - au_0
        if (b + 1) % 500 == 0:
            print(f"  ...{b+1}/{a.B}")

    # gia tri diem tren du lieu goc
    full = np.arange(N)
    pt_f1 = np.mean([f1_at(y, p, THR, full) for p in P1], axis=0) - \
            np.mean([f1_at(y, p, THR, full) for p in P0], axis=0)
    pt_au = np.nanmean([auprc_at(y, p, full) for p in P1], axis=0) - \
            np.nanmean([auprc_at(y, p, full) for p in P0], axis=0)

    rows = []
    for j, c in enumerate(CLASSES):
        lo1, hi1 = np.percentile(dF1[:, j], [2.5, 97.5])
        lo2, hi2 = np.percentile(dAU[:, j], [2.5, 97.5])
        rows.append(dict(tag=a.tag, cls=c, n_pos=int(y[:, j].sum()),
                         dF1=pt_f1[j], dF1_lo=lo1, dF1_hi=hi1,
                         dF1_excl0=bool(lo1 > 0 or hi1 < 0),
                         dAUPRC=pt_au[j], dAU_lo=lo2, dAU_hi=hi2,
                         dAU_excl0=bool(lo2 > 0 or hi2 < 0)))
    df = pd.DataFrame(rows)
    print("\n" + "=" * 88)
    print(f"E9 -- paired patient-level bootstrap ({a.B} lan) -- {a.tag}")
    print("Hieu = (input lead II) - (input lead I)")
    print("=" * 88)
    print(df.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print("""
dF1_excl0 = True  -> khoang tin cay 95% KHONG chua 0 -> chenh lech vung o muc BENH NHAN.
dF1_excl0 = False -> co the do co mau nho; bai phai noi ro dieu do.
""")
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    np.savez(outp.with_suffix(".npz"), dF1=dF1, dAUPRC=dAU, classes=CLASSES)
    print("Saved:", outp)


if __name__ == "__main__":
    main()
