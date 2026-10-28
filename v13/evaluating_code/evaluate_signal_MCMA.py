import argparse
import sys
from pathlib import Path

import torch


REPOSITORY_DIR = Path(__file__).resolve().parents[2]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, relative_path, require_repo_root
from v13.utils_MCMA import Trainer, make_dataloader, test_ae, write2excel

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MCMA checkpoint with PyTorch.")
    parser.add_argument("--testpath", type=relative_path, default="datasets/pt/ptbxl_testset.pt")
    parser.add_argument("--test2path", type=relative_path, default="datasets/pt/cpsc2018.pt")
    parser.add_argument("--modelpath", type=relative_path, default="checkpoints/v13/Autoencoder_ks_5.pt")
    parser.add_argument("--resultfile", type=artifact_path, default="artifacts/metrics/v13/ptbxl_diff_ks/Autoencoder_ks_5.xlsx")
    parser.add_argument("--resultfile2", type=artifact_path, default="artifacts/metrics/v13/cpsc2018_diff_ks/Autoencoder_ks_5.xlsx")
    parser.add_argument("--kernel_size", default=5, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--ecglen", default=1024, type=int)
    parser.add_argument("--pool_size", default=2, type=int)
    return parser.parse_args()


def main():
    require_repo_root()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    modelpath = Path(args.modelpath)
    if not modelpath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {modelpath}")

    test_loader = make_dataloader(
        args.testpath,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader2 = make_dataloader(
        args.test2path,
        batch_size=args.batch_size,
        shuffle=False,
    )

    trainer_ae = Trainer(
        modelpath=modelpath,
        ecglen=args.ecglen,
        kernel_size=args.kernel_size,
        pool_size=args.pool_size,
        device=device,
    )
    trainer_ae.load(modelpath)

    MSE_test, PCC_test = test_ae(
        model=trainer_ae.model,
        ds=test_loader,
        numlead=12,
        padding="zeros",
        device=device,
    )
    write2excel(MSE_test, PCC_test, args.resultfile)
    print("Saved:", args.resultfile)

    MSE_test, PCC_test = test_ae(
        model=trainer_ae.model,
        ds=test_loader2,
        numlead=12,
        padding="zeros",
        device=device,
    )
    write2excel(MSE_test, PCC_test, args.resultfile2)
    print("Saved:", args.resultfile2)

if __name__ == "__main__":
    main()
