import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from ptflops import get_model_complexity_info


REPOSITORY_DIR = Path(__file__).resolve().parents[2]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, relative_path, require_repo_root
from v13.utils_MCMA import Trainer, make_dataloader, test_ae, write2excel

def parse_args():
    parser = argparse.ArgumentParser(description="Train MCMA with PyTorch.")
    parser.add_argument("--trainpath", type=relative_path, default="datasets/pt/ptbxl_trainset.pt")
    parser.add_argument("--valpath", type=relative_path, default="datasets/pt/ptbxl_valset.pt")
    parser.add_argument("--testpath", type=relative_path, default="datasets/pt/ptbxl_testset.pt")
    parser.add_argument("--test2path", type=relative_path, default="datasets/pt/cpsc2018.pt")
    parser.add_argument("--modelpath", type=artifact_path, default="artifacts/checkpoints/v13/Autoencoder_ks_5.pt")
    parser.add_argument("--resultfile", type=artifact_path, default="artifacts/metrics/v13/ptbxl_diff_ws/Autoencoder_ws.xlsx")
    parser.add_argument("--resultfile2", type=artifact_path, default="artifacts/metrics/v13/cpsc2018_diff_ws/Autoencoder_ws.xlsx")
    parser.add_argument("--training", default=1, type=int, help="Set to 1 to train the model, 0 to skip training.")
    parser.add_argument("--kernel_size", default=5, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--ecglen", default=1024, type=int)
    parser.add_argument("--anylead", default=1, type=int)
    parser.add_argument("--patience", default=50, type=int)
    parser.add_argument("--figure_plot", default=0, type=int)
    return parser.parse_args()


def main():
    require_repo_root()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader = make_dataloader(args.trainpath, batch_size=args.batch_size, shuffle=True)
    val_loader = make_dataloader(args.valpath, batch_size=args.batch_size, shuffle=False)
    test_loader = make_dataloader(args.testpath, batch_size=args.batch_size, shuffle=False)
    test2_loader = make_dataloader(args.test2path, batch_size=args.batch_size, shuffle=False)

    modelpath = Path(args.modelpath)
    resultpath = Path(args.resultfile)
    resultpath2 = Path(args.resultfile2)

    resultpath.parent.mkdir(parents=True, exist_ok=True)
    resultpath2.parent.mkdir(parents=True, exist_ok=True)


    trainer_ae = Trainer(
        modelpath=modelpath,
        epochs=args.epochs,
        lr=args.lr,
        ecglen=args.ecglen,
        kernel_size=args.kernel_size,
        pool_size=2,              # hoặc truyền thành argparse nếu muốn
        figure_plot=args.figure_plot,
        step_total=len(train_loader),
        anylead=args.anylead,
        patience=args.patience,
        device=device,
    )

    if args.training:
        trainer_ae.train(train_loader, val_loader)

    model = trainer_ae.load(modelpath)

    try:
        flops, params = get_model_complexity_info(
            model,
            (12, args.ecglen),
            as_strings=False,
            print_per_layer_stat=False,
        )
        info = {
            "flops": flops / 1e6,
            "params": params / 1e6,
        }
    except Exception:
        flops, params = get_model_complexity_info(
            model,
            (12, args.ecglen),
            as_strings=True,
        )
        info = {
            "flops": flops,
            "params": params,
        }

    pd.DataFrame([info]).to_pickle("model_size.pkl")

    MSE_test, PCC_test = test_ae(
        model=model,
        ds=test_loader,
        numlead=12,
        padding="zeros",
        device=device,
    )
    write2excel(MSE_test, PCC_test, resultpath)
    print("Saved:", resultpath)

    MSE_test2, PCC_test2 = test_ae(
        model=model,
        ds=test2_loader,
        numlead=12,
        padding="zeros",
        device=device,
    )
    write2excel(MSE_test2, PCC_test2, resultpath2)
    print("Saved:", resultpath2)
if __name__ == "__main__":
    main()
