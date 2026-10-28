import os
import argparse
import sys
from pathlib import Path

import torch


REPOSITORY_DIR = Path(__file__).resolve().parents[2]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from project_paths import artifact_path, relative_path, require_repo_root
from experiment_metadata import (
    build_run_metadata,
    metadata_path_for,
    resolve_source_version,
    seed_everything,
    train_with_metadata,
)
from v13.utils_MCMA import Trainer, make_dataloader

VERSION = Path(__file__).resolve().parents[1].name

def parse_args():
    parser = argparse.ArgumentParser(description="Train MCMA with PyTorch.")
    parser.add_argument("--trainpath", type=relative_path, default="datasets/pt/ptbxl_trainset.pt")
    parser.add_argument("--valpath", type=relative_path, default="datasets/pt/ptbxl_valset.pt")
    parser.add_argument("--modelpath", type=artifact_path, default="artifacts/checkpoints/v13/Autoencoder_ks_5.pt")
    parser.add_argument("--kernel_size", default=5, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--ecglen", default=1024, type=int)
    parser.add_argument("--pool_size", default=2, type=int)
    parser.add_argument("--anylead", default=1, type=int)
    parser.add_argument("--patience", default=50, type=int)
    parser.add_argument("--figure_plot", default=0, type=int)
    parser.add_argument("--stage1_epochs", default=50, type=int) # Thêm mới
    parser.add_argument("--alpha", default=0.5, type=float) # Thêm mới
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--source-version", default=None)
    return parser.parse_args()


def main():
    repo_root = require_repo_root()
    args = parse_args()
    seed_everything(args.seed)
    source_version, source_dirty = resolve_source_version(
        repo_root, explicit=args.source_version
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader = make_dataloader(
        args.trainpath,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = make_dataloader(
        args.valpath,
        batch_size=args.batch_size,
        shuffle=False,
    )

    modelpath = Path(args.modelpath)
    modelpath.parent.mkdir(parents=True, exist_ok=True)

    trainer_ae = Trainer(
        modelpath=modelpath,
        epochs=args.epochs,
        lr=args.lr,
        ecglen=args.ecglen,
        kernel_size=args.kernel_size,
        pool_size=args.pool_size,
        figure_plot=args.figure_plot,
        step_total=len(train_loader),
        anylead=args.anylead,
        patience=args.patience,
        stage1_epochs=args.stage1_epochs,
        alpha=args.alpha,
        device=device,
    )
    metadata = build_run_metadata(
        args=args,
        version=VERSION,
        source_version=source_version,
        source_dirty=source_dirty,
        optimizer=trainer_ae.optimizer.__class__.__name__,
        device=str(device),
    )
    train_with_metadata(
        trainer_ae, train_loader, val_loader, metadata, metadata_path_for(modelpath)
    )


if __name__ == "__main__":
    main()
