"""
=====================================================================
 EXPERIMENT E0b -- Chieu truc giao len khong gian Einthoven-Goldberger
=====================================================================
KHONG CAN TRAIN LAI. Chay tren checkpoint DA CO (v1, v13, ...).
Day la cau tra loi cho van de "exact projection cho ket qua thap hon
baseline".

VAN DE CUA CACH LAM HIEN TAI (v14):
    Mang sinh 8 kenh [I,II,V1..V6], roi SUY RA 4 lead con lai.
    -> vut bo thong tin ma mang da hoc duoc o 4 kenh III/aVR/aVL/aVF
    -> sai so cua I va II CONG DON vao lead hieu (III = II - I)
    -> co the TE HON baseline. Da xac nhan bang mo phong.

CACH DUNG (E0b):
    Giu nguyen 12 kenh dau ra, roi CHIEU TRUC GIAO len khong gian hop le
        S = range(P),   P_S = P (P^T P)^-1 P^T
    Dung CA 12 kenh, khong vut gi ca.

DINH LY: neu ground truth y nam trong S thi voi moi du doan yhat
        || P_S(yhat) - y ||  <=  || yhat - y ||
Chung minh: y in S nen P_S(y)=y, do do
        P_S(yhat) - y = P_S(yhat - y), va projector truc giao la anh xa co.
=> Chieu truc giao KHONG BAO GIO lam tang MSE. Bang 0 khi mo hinh da
   thoa man dinh luat san; cang vi pham nhieu thi cang loi.

DIEU KIEN TIEN QUYET: ground truth phai thoa dinh luat. Chay
check_datasets.py truoc. Neu dataset chuan hoa RIENG tung lead thi
dinh ly khong ap dung.

CACH CHAY:
    python paper_v2/exp_E0b_ortho_projection.py ^
        --repo . --version v13 ^
        --modelpath <ckpt> ^
        --testpath datasets\pt\ptbxl_testset.pt ^
        --out paper_v2\results_E\E0b_v13_ptbxl.csv
=====================================================================
"""
import argparse, importlib, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
IND   = [0,1,6,7,8,9,10,11]          # I, II, V1..V6
DERIV = [2,3,4,5]                    # III, aVR, aVL, aVF


def build_P(dtype=torch.float32):
    P = torch.zeros(12, 8, dtype=dtype)
    P[0,0]=1.0; P[1,1]=1.0
    P[2,0],P[2,1] = -1.0, 1.0        # III = II - I
    P[3,0],P[3,1] = -0.5,-0.5        # aVR = -(I+II)/2
    P[4,0],P[4,1] =  1.0,-0.5        # aVL = I - II/2
    P[5,0],P[5,1] = -0.5, 1.0        # aVF = II - I/2
    for j in range(6): P[6+j, 2+j] = 1.0
    return P


def build_PS(P):
    """Projector truc giao len range(P): P (P^T P)^-1 P^T, kich thuoc 12x12."""
    return P @ torch.linalg.inv(P.T @ P) @ P.T


def pcc(a, b, eps=1e-8):
    ac = a - a.mean(-1, keepdim=True); bc = b - b.mean(-1, keepdim=True)
    num = (ac*bc).mean(-1)
    den = a.std(-1, unbiased=False)*b.std(-1, unbiased=False)
    return torch.nan_to_num(num/(den+eps), nan=0.0, posinf=0.0, neginf=0.0)


def fwd(model, x):
    o = model(x)
    return o[0] if isinstance(o, (tuple, list)) else o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--version", default="v13")
    ap.add_argument("--modelpath", required=True)
    ap.add_argument("--testpath", required=True)
    ap.add_argument("--out", default="E0b_results.csv")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--ecglen", type=int, default=1024)
    ap.add_argument("--kernel_size", type=int, default=5)
    ap.add_argument("--pool_size", type=int, default=2)
    a = ap.parse_args()

    repo = str(Path(a.repo).resolve())
    if repo not in sys.path: sys.path.insert(0, repo)
    U = importlib.import_module(f"{a.version}.utils_MCMA")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P  = build_P().to(dev)
    PS = build_PS(build_P()).to(dev)
    print(f"Device {dev} | version {a.version}")
    print(f"kiem tra projector: ||PS@PS-PS|| = {torch.norm(PS@PS-PS):.2e}, "
          f"rank = {torch.linalg.matrix_rank(PS).item()} (phai = 8)\n")

    loader = U.make_dataloader(a.testpath, batch_size=a.batch_size, shuffle=False)
    tr = U.Trainer(modelpath=Path(a.modelpath), ecglen=a.ecglen,
                   kernel_size=a.kernel_size, pool_size=a.pool_size, device=dev)
    model = tr.load(Path(a.modelpath)).to(dev).eval()

    n = 0
    METH = ["raw", "derive8", "ortho"]
    mse = {m: np.zeros((12,12)) for m in METH}   # [input lead, output lead]
    pc  = {m: np.zeros((12,12)) for m in METH}
    gt_viol = 0.0

    with torch.no_grad():
        for step, batch in enumerate(loader):
            y = U.ecg_to_channels_first(batch).to(dev)
            y = y[y.std(-1).mean(-1) >= 1e-4]
            if y.numel() == 0: continue
            n += y.size(0)

            # ground truth co thoa dinh luat khong? (dieu kien cua dinh ly)
            gt_viol += (torch.matmul(PS.unsqueeze(0), y) - y).pow(2).mean((1,2)).sum().item()

            for k in range(12):
                idx = torch.full((y.size(0),), k, dtype=torch.long, device=dev)
                yh = fwd(model, U.paddingecg(y, idx))
                out = {
                    "raw":     yh,
                    "derive8": torch.matmul(P.unsqueeze(0),  yh[:, IND, :]),
                    "ortho":   torch.matmul(PS.unsqueeze(0), yh),
                }
                for m, z in out.items():
                    mse[m][k] += (z-y).pow(2).mean(-1).sum(0).cpu().numpy()
                    pc[m][k]  += pcc(y, z).sum(0).cpu().numpy()
            if step % 5 == 0: print(f"  step {step}, n={n}")

    for m in METH:
        mse[m] /= max(n,1); pc[m] /= max(n,1)
    gt_viol /= max(n,1)

    off = ~np.eye(12, dtype=bool)
    print("\n" + "="*78)
    print(f"E0b -- {a.version} -- {Path(a.testpath).name} -- n={n}")
    print("="*78)
    print(f"Vi pham dinh luat tren GROUND TRUTH: {gt_viol:.3e}")
    print("  (phai ~0. Neu lon => dataset chuan hoa rieng tung lead,")
    print("   dinh ly khong ap dung, DUNG LAI bao thay.)\n")

    rows = []
    print(f"{'lead':6s}" + "".join(f"{m+'_PCC':>13s}{m+'_MSE':>13s}" for m in METH))
    for j, nm in enumerate(LEADS):
        r = {"lead": nm, "is_derived": j in DERIV}
        line = f"{nm:6s}"
        for m in METH:
            p_ = pc[m][off[:,j], j].mean(); e_ = mse[m][off[:,j], j].mean()
            r[f"PCC_{m}"] = p_; r[f"MSE_{m}"] = e_
            line += f"{p_:13.4f}{e_:13.5f}"
        rows.append(r); print(line)

    tot = {"lead": "ALL(off-diag)", "is_derived": ""}
    line = f"{'ALL':6s}"
    for m in METH:
        p_ = pc[m][off].mean(); e_ = mse[m][off].mean()
        tot[f"PCC_{m}"] = p_; tot[f"MSE_{m}"] = e_
        line += f"{p_:13.4f}{e_:13.5f}"
    rows.append(tot); print(line)

    df = pd.DataFrame(rows)
    d = df[df.is_derived == True]
    print("\n--- Chi 4 lead dan xuat (III, aVR, aVL, aVF) ---")
    for m in METH:
        print(f"  {m:8s} PCC={d[f'PCC_{m}'].mean():.4f}  MSE={d[f'MSE_{m}'].mean():.5f}")

    print(f"""
DOC KET QUA:
  ortho vs raw  : MSE cua 'ortho' PHAI <= 'raw' (dinh ly). Neu khong,
                  ground truth khong thoa dinh luat -> xem dong vi pham o tren.
  derive8 vs raw: day la cach v14 dang lam. Neu te hon 'raw' thi da giai
                  thich vi sao v14 thap hon baseline -- va 'ortho' la ban va.
  Muc loi cua ortho ty le thuan voi muc mo hinh vi pham dinh luat (E0).
""")
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    np.savez(outp.with_suffix(".npz"), n=n, gt_viol=gt_viol,
             **{f"mse_{m}": mse[m] for m in METH}, **{f"pcc_{m}": pc[m] for m in METH})
    print("Saved:", outp)


if __name__ == "__main__":
    main()
