"""
Chuyen file .npy do save_genecg.py sinh ra thanh .hdf5 dung dinh dang ma
predict.py cua Ribeiro (automatic-ecg-diagnosis) doc duoc.

save_genecg.py  -> (N, 4096, 12) float32 .npy
predict.py      -> can .hdf5 co dataset ten "tracings", shape (N, 4096, 12)

CACH CHAY TU REPO ROOT:
    python paper_v2/npy_to_hdf5.py `
        --inp artifacts/generated_ecg/gen_v13_lead0.npy `
        --out artifacts/generated_ecg/gen_v13_lead0.hdf5
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, relative_path, require_repo_root


def main():
    require_repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=relative_path, required=True)
    ap.add_argument("--out", type=artifact_path, required=True)
    ap.add_argument("--key", default="tracings")
    args = ap.parse_args()

    x = np.load(args.inp, allow_pickle=False)
    if x.ndim != 3:
        raise ValueError(f"Expected 3-D array, got {x.shape}")
    if x.shape[1] == 12 and x.shape[2] != 12:
        print(f"  transpose {x.shape} -> (N, L, 12)")
        x = np.transpose(x, (0, 2, 1))
    if x.shape[2] != 12:
        raise ValueError(f"Last axis must be 12 leads, got {x.shape}")

    print(f"  shape {x.shape}, dtype {x.dtype}, "
          f"min {x.min():.4f}, max {x.max():.4f}")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(outp, "w") as f:
        f.create_dataset(args.key, data=x.astype(np.float32))
    print("Saved:", outp)


if __name__ == "__main__":
    main()
