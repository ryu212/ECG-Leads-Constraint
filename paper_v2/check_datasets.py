"""
=====================================================================
 CHECK DATASETS -- kiem tra cac file .pt truoc khi chay bat cu gi
=====================================================================
Repo KHONG co script tao cac file .pt. Chung phai ton tai san. Script
nay kiem tra chung dung dinh dang ma utils_MCMA mong doi, va bao cao
thong ke de phat hien sai chuan hoa / sai tan so lay mau.

Dinh dang bat buoc:
    - torch.save mot tensor (N, L, 12) HOAC mot dict {"ecg": tensor}
    - L = 1024 (mac dinh), 12 lead o TRUC CUOI
    - thu tu lead: I, II, III, aVR, aVL, aVF, V1..V6

CACH CHAY TU REPO ROOT:
    python paper_v2/check_datasets.py datasets/pt/ptbxl_trainset.pt \
        datasets/pt/ptbxl_valset.pt datasets/pt/ptbxl_testset.pt \
        datasets/pt/cpsc2018.pt
=====================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import relative_path, require_repo_root

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]


def check(path):
    p = Path(path)
    print("\n" + "=" * 66)
    print(p.name)
    print("=" * 66)
    if not p.exists():
        print("  !! KHONG TON TAI")
        return

    obj = torch.load(p, map_location="cpu")
    data = obj["ecg"] if isinstance(obj, dict) else obj
    if isinstance(obj, dict):
        print("  kieu           : dict, keys =", list(obj.keys()))
    else:
        print("  kieu           : tensor truc tiep")

    data = data.float()
    print("  shape          :", tuple(data.shape), " (mong doi: N, L, 12)")

    ok = True
    if data.ndim != 3:
        print("  !! LOI: phai la tensor 3 chieu")
        return
    if data.shape[2] != 12:
        print(f"  !! LOI: truc cuoi phai la 12 lead, dang la {data.shape[2]}")
        print("     Neu shape la (N,12,L) thi phai transpose truoc khi luu,")
        print("     vi ecg_to_channels_first() se permute(0,2,1).")
        ok = False
    if data.shape[1] != 1024:
        print(f"  !  CANH BAO: do dai {data.shape[1]}, mac dinh ecglen=1024")

    print(f"  so ban ghi     : {data.shape[0]}")
    print(f"  gia tri        : min={data.min():.4f}  max={data.max():.4f}  "
          f"mean={data.mean():.4f}  std={data.std():.4f}")

    flat_std = data.std(dim=1)                       # (N,12)
    dead = (flat_std.mean(dim=1) < 1e-4).sum().item()
    print(f"  ban ghi phang  : {dead} (se bi test_ae loai bo)")
    nan = torch.isnan(data).sum().item()
    inf = torch.isinf(data).sum().item()
    if nan or inf:
        print(f"  !! LOI: NaN={nan}, Inf={inf}")
        ok = False

    # Kiem tra dinh luat Einthoven tren DU LIEU THAT.
    # Neu du lieu that cung vi pham nang thi van de nam o tien xu ly,
    # khong phai o mo hinh -- phai biet dieu nay TRUOC khi viet bai.
    x = data[: min(2000, data.shape[0])]
    i, ii, iii = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    avr, avl, avf = x[:, :, 3], x[:, :, 4], x[:, :, 5]
    def rel(a, b):
        num = (a - b).pow(2).mean(dim=1).sqrt()
        den = a.pow(2).mean(dim=1).sqrt() + 1e-8
        return (num / den).mean().item()
    print("  --- dinh luat tren DU LIEU THAT (cang gan 0 cang tot) ---")
    print(f"    III vs (II - I)       : {rel(iii, ii - i):.5f}")
    print(f"    aVR vs -(I + II)/2    : {rel(avr, -0.5 * (i + ii)):.5f}")
    print(f"    aVL vs (I - II/2)     : {rel(avl, i - 0.5 * ii):.5f}")
    print(f"    aVF vs (II - I/2)     : {rel(avf, ii - 0.5 * i):.5f}")
    print("    (neu cac so nay > 0.05 thi dataset da bi chuan hoa RIENG")
    print("     tung lead -- dinh luat khong con dung, va lop chieu exact")
    print("     SE KHONG AP DUNG DUOC. Phai chuan hoa chung cho ca 12 lead.)")

    print("  --- std tung lead ---")
    per_lead = data.std(dim=(0, 1)).numpy()
    print("    " + "  ".join(f"{n}={v:.3f}" for n, v in zip(LEADS, per_lead)))

    print("  KET LUAN       :", "OK" if ok else "CAN SUA")


def main():
    require_repo_root()
    parser = argparse.ArgumentParser(description="Validate ECG .pt datasets.")
    parser.add_argument("paths", nargs="+", type=relative_path)
    args = parser.parse_args()
    for path in args.paths:
        check(path)


if __name__ == "__main__":
    main()
