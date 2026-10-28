"""
=====================================================================
 EXPERIMENT E3 -- Nang luong non-dipolar E_nd
=====================================================================
KHONG CAN TRAIN LAI (chi can checkpoint da co).

Do truc tiep phan tin hieu ma phep chieu VCG KHONG THE bieu dien:

    E_nd(y8) = || (I - K^+ K) y8 ||^2

trong do K la ma tran Kors (3x8) va K^+ la gia nghich dao Moore-Penrose.
(I - K^+K) chieu len khong gian null cua K -- dung phan ma L_VCG mu.

GIA THUYET can kiem chung:
    Khi alpha (trong so VCG loss) tang, E_nd cua tin hieu TAI TAO giam
    dung o nhung lead ma hieu nang chan doan giam (Lead II).
Neu dung => day la CO CHE NHAN QUA, khong con la suy doan.

---------------------------------------------------------------------
CACH CHAY TU REPO ROOT
---------------------------------------------------------------------
    python paper_v2/exp_E3_nondipolar.py ^
        --version   v13 ^
        --modelpath checkpoints/v13/Autoencoder_ks_5.pt ^
        --testpath  datasets/pt/ptbxl_testset.pt ^
        --tag       alpha0.5 ^
        --out       artifacts/metrics/E3/E3_alpha0.5_ptbxl.csv

Chay lai voi tung checkpoint alpha = 0 / 0.1 / 0.5 / 1.0 (thi nghiem E2),
doi --tag moi lan, roi gop cac file CSV lai de ve Figure 1 (fig:nondip).
=====================================================================
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, relative_path, require_repo_root

# 8 lead doc lap theo dung thu tu cua lead_idx trong loss.py
IND_IDX = [0, 1, 6, 7, 8, 9, 10, 11]
IND_NAMES = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
ALL_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]

# Ma tran Kors, giong het trong v13/loss/loss.py
KORS = np.array([
    [-0.130,  0.050, -0.010,  0.140,  0.060,  0.540,  0.380, -0.070],  # X
    [ 0.060, -0.020, -0.050,  0.060, -0.170,  0.130, -0.070,  0.930],  # Y
    [-0.430, -0.060, -0.140, -0.200, -0.110,  0.310,  0.110, -0.230],  # Z
], dtype=np.float64)
# LUU Y: thu tu cot cua KORS trong loss.py la [I, II, V1..V6] -- khop IND_IDX.


def null_projector(K):
    """Tra ve (I - K^+ K), ma tran 8x8 chieu len null space cua K."""
    Kp = np.linalg.pinv(K)                 # 8x3
    return np.eye(K.shape[1]) - Kp @ K     # 8x8


def forward_any(model, x):
    out = model(x)
    return out[0] if isinstance(out, (tuple, list)) else out


def main():
    repo = require_repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v13")
    ap.add_argument("--modelpath", type=relative_path, required=True)
    ap.add_argument("--testpath", type=relative_path, required=True)
    ap.add_argument("--tag", default="run",
                    help="Nhan de phan biet cac alpha, vd alpha0.5")
    ap.add_argument(
        "--out", type=artifact_path, default="artifacts/metrics/E3/E3_results.csv"
    )
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--ecglen", type=int, default=1024)
    ap.add_argument("--kernel_size", type=int, default=5)
    ap.add_argument("--pool_size", type=int, default=2)
    args = ap.parse_args()

    repo = str(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    utils = importlib.import_module(f"{args.version}.utils_MCMA")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | version: {args.version} | tag: {args.tag}")

    N = torch.tensor(null_projector(KORS), dtype=torch.float32, device=device)
    print("rank(K) =", np.linalg.matrix_rank(KORS),
          "| rank(I - K^+K) =", np.linalg.matrix_rank(null_projector(KORS)))

    loader = utils.make_dataloader(args.testpath,
                                   batch_size=args.batch_size, shuffle=False)
    trainer = utils.Trainer(
        modelpath=Path(args.modelpath), ecglen=args.ecglen,
        kernel_size=args.kernel_size, pool_size=args.pool_size, device=device,
    )
    model = trainer.load(Path(args.modelpath)).to(device).eval()

    n = 0
    e_true = np.zeros(8)              # E_nd cua tin hieu that
    e_pred = np.zeros((12, 8))        # E_nd cua tai tao, theo input lead

    with torch.no_grad():
        for step, batch in enumerate(loader):
            y = utils.ecg_to_channels_first(batch).to(device)
            keep = y.std(dim=-1).mean(dim=-1) >= 1e-4
            y = y[keep]
            if y.numel() == 0:
                continue
            n += y.size(0)

            y8 = y[:, IND_IDX, :]                       # (B,8,L)
            nd_true = torch.matmul(N.unsqueeze(0), y8)  # (B,8,L)
            e_true += nd_true.pow(2).mean(dim=-1).sum(dim=0).cpu().numpy()

            for k in range(12):
                idx = torch.full((y.size(0),), k, dtype=torch.long, device=device)
                y_hat = forward_any(model, utils.paddingecg(y, idx))
                nd_pred = torch.matmul(N.unsqueeze(0), y_hat[:, IND_IDX, :])
                e_pred[k] += nd_pred.pow(2).mean(dim=-1).sum(dim=0).cpu().numpy()

            if step % 5 == 0:
                print(f"  step {step}, n={n}")

    e_true /= max(n, 1)
    e_pred /= max(n, 1)

    off = ~np.eye(12, dtype=bool)
    rows = []
    for j, name in enumerate(IND_NAMES):
        col = IND_IDX[j]
        m = off[:, col]
        et, ep = e_true[j], e_pred[m, j].mean()
        rows.append({
            "tag": args.tag,
            "lead": name,
            "E_nd_true": et,
            "E_nd_pred": ep,
            "ratio_pred_over_true": ep / (et + 1e-12),
        })
    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print(f"E3 -- non-dipolar residual energy -- {args.tag}")
    print("=" * 70)
    print(df.to_string(index=False, float_format=lambda v: f"{v:10.6f}"))
    print("""
ratio < 1  => tai tao da MAT thanh phan non-dipolar so voi tin hieu that.
Ky vong: ty le nay GIAM khi alpha tang, va giam manh nhat o Lead II.
""")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    np.savez(outp.with_suffix(".npz"), e_true=e_true, e_pred=e_pred, n=n)
    print("Saved:", outp)


if __name__ == "__main__":
    main()
