"""
=====================================================================
 EXPERIMENT E0 -- Kiem tra vi pham dinh luat Einthoven / Goldberger
=====================================================================
KHONG CAN TRAIN LAI. Chay tren checkpoint da co, mat vai phut.

Script nay tra ve HAI con so quyet dinh cho dong gop so 1 cua bai bao:

 (a) violation_norm -- mo hinh hien tai lech bao nhieu so voi cac dong
     nhat thuc ma no chua bao gio duoc day:
         III  = II - I
         aVR  = -(I + II)/2
         aVL  = I - II/2
         aVF  = II - I/2

 (b) PCC_derived vs PCC_direct -- neu ta SUY RA cac lead chi tu I va II
     du doan, ket qua co tot hon lead ma mang du doan truc tiep khong?
     So do hien tai (PTB-XL, v1): III=0.520, aVR=0.869, aVL=0.674, aVF=0.636
     Neu delta > 0 => lap luan "exact projection" duoc chung minh
     TRUOC khi ton mot gio GPU nao.

---------------------------------------------------------------------
CACH CHAY TU REPO ROOT
---------------------------------------------------------------------
    python paper_v2/exp_E0_einthoven_check.py ^
        --version   v1 ^
        --modelpath checkpoints/v1/Autoencoder_ks_5.pth ^
        --testpath  datasets/pt/ptbxl_testset.pt ^
        --out       artifacts/metrics/E0/E0_v1_ptbxl.csv

Chay lai voi --testpath cpsc2018.pt de co cot CPSC2018.
Chay lai voi --version v13 --modelpath <ckpt v13> de so sanh hai mo hinh.

Thu tu lead giong utils_MCMA:
    0=I 1=II 2=III 3=aVR 4=aVL 5=aVF 6=V1 7=V2 8=V3 9=V4 10=V5 11=V6
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
from tools.experiment_common import write_evaluation_metadata

LEAD_I, LEAD_II = 0, 1
DERIVED_IDX = [2, 3, 4, 5]          # III, aVR, aVL, aVF
DERIVED_NAMES = ["III", "aVR", "aVL", "aVF"]


def derive_limb(y):
    """Suy ra III, aVR, aVL, aVF tu lead I va II.

    y : (B, 12, L). Tra ve (B, 4, L) theo thu tu [III, aVR, aVL, aVF].
    """
    li, lii = y[:, LEAD_I, :], y[:, LEAD_II, :]
    return torch.stack([
        lii - li,                 # III
        -0.5 * (li + lii),        # aVR
        li - 0.5 * lii,           # aVL
        lii - 0.5 * li,           # aVF
    ], dim=1)


def pcc(a, b, eps=1e-8):
    """Pearson correlation theo truc cuoi. a, b : (B, C, L) -> (B, C)."""
    ac = a - a.mean(dim=-1, keepdim=True)
    bc = b - b.mean(dim=-1, keepdim=True)
    num = (ac * bc).mean(dim=-1)
    den = a.std(dim=-1, unbiased=False) * b.std(dim=-1, unbiased=False)
    return torch.nan_to_num(num / (den + eps), nan=0.0, posinf=0.0, neginf=0.0)


def forward_any(model, x):
    """v1..v7 tra ve 1 tensor; v8/v12/v13 tra ve (pred, base)."""
    out = model(x)
    return out[0] if isinstance(out, (tuple, list)) else out


def main():
    repo = require_repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1",
                    help="v1 | v2 | v3 | v4 | v7 | v8 | v12 | v13")
    ap.add_argument("--modelpath", type=relative_path, required=True)
    ap.add_argument("--testpath", type=relative_path, required=True)
    ap.add_argument(
        "--out", type=artifact_path, default="artifacts/metrics/E0/E0_results.csv"
    )
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--ecglen", type=int, default=1024)
    ap.add_argument("--kernel_size", type=int, default=5)
    ap.add_argument("--pool_size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None,
                    help="Optional checkpoint seed; omit for legacy checkpoints.")
    args = ap.parse_args()

    repo = str(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    utils = importlib.import_module(f"{args.version}.utils_MCMA")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | version: {args.version}")

    loader = utils.make_dataloader(args.testpath,
                                   batch_size=args.batch_size, shuffle=False)

    trainer = utils.Trainer(
        modelpath=Path(args.modelpath),
        ecglen=args.ecglen,
        kernel_size=args.kernel_size,
        pool_size=args.pool_size,
        device=device,
    )
    model = trainer.load(Path(args.modelpath)).to(device).eval()

    n = 0
    violation = np.zeros((12, 4))     # [input lead, derived lead]
    pcc_direct = np.zeros((12, 4))
    pcc_derived = np.zeros((12, 4))

    with torch.no_grad():
        for step, batch in enumerate(loader):
            y = utils.ecg_to_channels_first(batch).to(device)
            keep = y.std(dim=-1).mean(dim=-1) >= 1e-4
            y = y[keep]
            if y.numel() == 0:
                continue
            n += y.size(0)

            true_limb = y[:, DERIVED_IDX, :]
            rms_true = true_limb.pow(2).mean(dim=-1).sqrt() + 1e-8

            for k in range(12):
                idx = torch.full((y.size(0),), k, dtype=torch.long, device=device)
                y_hat = forward_any(model, utils.paddingecg(y, idx))

                pred_limb = y_hat[:, DERIVED_IDX, :]
                der_limb = derive_limb(y_hat)

                resid = (pred_limb - der_limb).pow(2).mean(dim=-1).sqrt()
                violation[k] += (resid / rms_true).sum(dim=0).cpu().numpy()
                pcc_direct[k] += pcc(true_limb, pred_limb).sum(dim=0).cpu().numpy()
                pcc_derived[k] += pcc(true_limb, der_limb).sum(dim=0).cpu().numpy()

            if step % 5 == 0:
                print(f"  step {step}, n={n}")

    violation /= max(n, 1)
    pcc_direct /= max(n, 1)
    pcc_derived /= max(n, 1)

    off = ~np.eye(12, dtype=bool)
    rows = []
    for j, name in enumerate(DERIVED_NAMES):
        m = off[:, DERIVED_IDX[j]]     # bo truong hop input chinh la lead do
        rows.append({
            "lead": name,
            "violation_norm": violation[m, j].mean(),
            "PCC_direct": pcc_direct[m, j].mean(),
            "PCC_derived": pcc_derived[m, j].mean(),
            "delta": pcc_derived[m, j].mean() - pcc_direct[m, j].mean(),
        })
    df = pd.DataFrame(rows)

    print("\n" + "=" * 64)
    print(f"E0 RESULTS -- {args.version} -- {Path(args.testpath).name}")
    print("(trung binh tren cac input lead, bo truong hop input = lead do)")
    print("=" * 64)
    print(df.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print("""
violation_norm = RMS(lead_du_doan - lead_suy_ra) / RMS(lead_that)
    ~ 0     => mo hinh da tu thoa man dong nhat thuc (khong ky vong)
    lon     => mo hinh dang vi pham mot dinh luat chinh xac

delta = PCC_derived - PCC_direct
    > 0     => LEAD SUY RA CHINH XAC HON lead mang du doan truc tiep.
               Day la bang chung cho dong gop so 1, khong can train lai.
""")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    np.savez(outp.with_suffix(".npz"), violation=violation,
             pcc_direct=pcc_direct, pcc_derived=pcc_derived, n=n)
    write_evaluation_metadata(
        outp.with_name(f"{outp.stem}_run_metadata.csv"),
        version=args.version,
        modelpath=args.modelpath,
        dataset=args.testpath,
        tag=outp.stem,
        extra={
            "seed": "" if args.seed is None else args.seed,
            "records": n,
            "batch_size": args.batch_size,
            "ecglen": args.ecglen,
            "kernel_size": args.kernel_size,
            "pool_size": args.pool_size,
        },
    )
    print("Saved:", outp)
    print("Saved:", outp.with_suffix(".npz"), "(ma tran 12x4 day du de ve Figure 1)")
    print("Saved:", outp.with_name(f"{outp.stem}_run_metadata.csv"))


if __name__ == "__main__":
    main()
