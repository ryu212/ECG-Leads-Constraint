"""
=====================================================================
 DANH GIA MUC CHAN DOAN (diagnostic level) tren CODE-test
=====================================================================
Dung cho thi nghiem E2 va E5 -- day la KET QUA QUAN TRONG NHAT con
thieu trong bai bao.

DAU VAO:
  --pred   file .npy xac suat do predict.py cua Ribeiro sinh ra,
           shape (827, 6)
  --gold   gold_standard.csv trong CODE-test (827 dong x 6 cot)

6 benh, dung thu tu cua Ribeiro:
  1dAVb, RBBB, LBBB, SB, AF, ST

DAU RA: precision / recall / F1 / AUPRC cho tung benh va macro-average,
o hai che do nguong:
  (a) nguong co dinh 0.5           -- bao cao chinh, khach quan
  (b) nguong toi uu F1 tren chinh tap nay -- LAC QUAN, chi de tham khao,
      PHAI ghi ro trong bai neu dung
AUPRC khong phu thuoc nguong, nen dung lam chi so chinh khi so sanh
giua cac alpha.

CACH CHAY TU REPO ROOT:
    python paper_v2/eval_diagnostic.py `
        --pred artifacts/predictions/pred_v13_lead0.npy `
        --gold external/code_test/gold_standard.csv `
        --tag "v13_alpha0.5_leadI" `
        --out artifacts/metrics/E2/E2_scores.csv
Moi lan chay se APPEND mot dong vao file --out, nen chay lan luot cho
tung alpha va tung input lead roi mo file do ra la co ca bang.
=====================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_fscore_support)

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, relative_path, require_repo_root

CLASSES = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]


def scores_at(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=[0, 1]
    )
    # lay lop duong tinh
    return p[1], r[1], f[1]


def best_f1_threshold(y_true, y_prob):
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        _, _, f = scores_at(y_true, y_prob, t)
        if f > best_f:
            best_f, best_t = f, t
    return best_t


def main():
    require_repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=relative_path, required=True)
    ap.add_argument("--gold", type=relative_path, required=True)
    ap.add_argument("--tag", required=True,
                    help='Nhan, vd "v13_alpha0.5_leadII"')
    ap.add_argument(
        "--out", type=artifact_path, default="artifacts/metrics/E2/E2_scores.csv"
    )
    args = ap.parse_args()

    prob = np.load(args.pred)
    if prob.ndim == 3:          # predict.py doi khi tra ve (1, N, 6)
        prob = prob[0]
    gold = pd.read_csv(args.gold)
    y = gold[CLASSES].values if all(c in gold.columns for c in CLASSES) \
        else gold.values
    y = np.asarray(y).astype(int)

    if prob.shape != y.shape:
        raise ValueError(f"pred {prob.shape} != gold {y.shape}")

    rows = []
    for j, c in enumerate(CLASSES):
        yt, yp = y[:, j], prob[:, j]
        p50, r50, f50 = scores_at(yt, yp, 0.5)
        tb = best_f1_threshold(yt, yp)
        pb, rb, fb = scores_at(yt, yp, tb)
        auprc = average_precision_score(yt, yp) if yt.sum() > 0 else np.nan
        rows.append({
            "tag": args.tag, "class": c, "n_pos": int(yt.sum()),
            "P@0.5": p50, "R@0.5": r50, "F1@0.5": f50,
            "best_thr": tb, "P@best": pb, "R@best": rb, "F1@best": fb,
            "AUPRC": auprc,
        })

    df = pd.DataFrame(rows)
    macro = {
        "tag": args.tag, "class": "MACRO", "n_pos": int(y.sum()),
        "P@0.5": df["P@0.5"].mean(), "R@0.5": df["R@0.5"].mean(),
        "F1@0.5": df["F1@0.5"].mean(), "best_thr": np.nan,
        "P@best": df["P@best"].mean(), "R@best": df["R@best"].mean(),
        "F1@best": df["F1@best"].mean(), "AUPRC": df["AUPRC"].mean(),
    }
    df = pd.concat([df, pd.DataFrame([macro])], ignore_index=True)

    print("\n" + "=" * 78)
    print(f"DIAGNOSTIC SCORES -- {args.tag}")
    print("=" * 78)
    print(df.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))
    print("""
Dung F1@0.5 va AUPRC lam so bao cao chinh.
F1@best dung nguong toi uu tren chinh tap test -> LAC QUAN, neu dua vao
bai phai ghi ro la "threshold tuned on the test set".
Cot quan trong nhat cho lap luan cua bai la PRECISION: nhanh residual
phai tang recall MA KHONG lam sut precision.
""")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    header = not outp.exists()
    df.to_csv(outp, mode="a", header=header, index=False)
    print("Appended to:", outp)


if __name__ == "__main__":
    main()
