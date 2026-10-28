from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from experiment_metadata import checkpoint_metadata_fields
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from .loss.loss import VCGConstraintLoss
from .model import MCMA_model

# load file .pt 
class ECGPTDataset(Dataset):
    def __init__(self, pt_file):
        obj = torch.load(pt_file, map_location="cpu")
        self.data = obj["ecg"] if isinstance(obj, dict) else obj
        self.metadata = obj.get("metadata", {}) if isinstance(obj, dict) else {}
        self.data = self.data.float()
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        return self.data[index]

def read_pt(pt_file):
    return ECGPTDataset(pt_file)

def make_dataloader(pt_file, batch_size=256, shuffle=False, num_workers=0):
    dataset = read_pt(pt_file)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def ecg_to_channels_first(ecg12):
    return ecg12.permute(0, 2, 1).contiguous()


def paddingecg(ecg12, lead_index):
    masked = torch.zeros_like(ecg12)
    batch_index = torch.arange(ecg12.size(0), device=ecg12.device)
    masked[batch_index, lead_index, :] = ecg12[batch_index, lead_index, :]
    return masked

# calculate pearson correlation coefficient
def get_pcc(ecg12, gen_ecg12, eps=1e-8):
    ecg_mean = ecg12.mean(dim=-1)
    gen_mean = gen_ecg12.mean(dim=-1)
    numerator = (ecg12 * gen_ecg12).mean(dim=-1) - ecg_mean * gen_mean
    denominator = (ecg12.std(dim=-1, unbiased=False) * gen_ecg12.std(dim=-1, unbiased=False))
    return numerator / (denominator + eps)


def compute_metric(gen_ecg12, ecg12):
    mse = torch.mean((gen_ecg12 - ecg12) ** 2, dim=-1).sum(dim=0)
    pcc_item = get_pcc(ecg12, gen_ecg12)
    pcc_item = torch.nan_to_num(pcc_item, nan=0.0, posinf=0.0, neginf=0.0)
    pcc = pcc_item.sum(dim=0)
    return mse.cpu().numpy(), pcc.cpu().numpy()


class Trainer:
    def __init__(
        self,
        modelpath,
        epochs=100,
        lr=1e-3,
        ecglen=1024,
        kernel_size=5,
        pool_size=2,
        figure_plot=0,
        step_total=340,
        anylead=1,
        patience=50,
        device=None,
        stage1_epochs=50,
        alpha=0.5
    ):
        self.ecglen = ecglen
        self.epochs = epochs
        self.lr = lr
        self.figure_plot = figure_plot
        self.kernel_size = kernel_size
        self.anylead = anylead
        self.step_total = step_total
        self.patience = patience
        self.pool_size = pool_size
        self.modelpath = modelpath
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.stage1_epochs = stage1_epochs
        self.model = MCMA_model(
            input_size=(self.ecglen, 12),
            kernel_size=self.kernel_size,
            pool_size=self.pool_size,
        ).to(self.device)
        self.criterion = VCGConstraintLoss(device=self.device, alpha= alpha)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.best_loss1 = float("inf")
        self.waiting = 0
    # according aythor' code, index 1 (which is lead II)is fixed when anylead is false  
    def _make_input(self, ecg12):
        batch_size = ecg12.size(0)
        if self.anylead == 1:
            lead_index = torch.randint(0, 12, (batch_size,), device=ecg12.device)
        else:
            lead_index = torch.zeros(batch_size, dtype=torch.long, device=ecg12.device)
        return paddingecg(ecg12, lead_index)

    def train_step(self, ecg12, is_stage1):
        self.model.train()
        ecg12 = ecg_to_channels_first(ecg12).to(self.device)
        ecg1 = self._make_input(ecg12)

        # unpack 2 đầu ra
        gen_ecg12, base_ecg12 = self.model(ecg1) 
        if is_stage1:
            # Stage 1: Pred = Base. Hàm Loss chỉ phạt trên Base
            loss = self.criterion(base_ecg12, ecg12, base_ecg12)
            out_ecg = base_ecg12
        else:
            # Stage 2: Pred = Base + Residual. VCG Loss vẫn chỉ phạt Base
            loss = self.criterion(gen_ecg12, ecg12, base_ecg12)
            out_ecg = gen_ecg12

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            pcc = get_pcc(ecg12, out_ecg)
        return loss.item(), torch.nan_to_num(pcc).mean().item()

    @torch.no_grad()
    def test_step(self, ecg12, is_stage1):
        self.model.eval()
        ecg12 = ecg_to_channels_first(ecg12).to(self.device)
        ecg1 = self._make_input(ecg12)
        gen_ecg12, base_ecg12 = self.model(ecg1)
        
        if is_stage1:
            loss = self.criterion(base_ecg12, ecg12, base_ecg12)
            out_ecg = base_ecg12
        else:
            loss = self.criterion(gen_ecg12, ecg12, base_ecg12)
            out_ecg = gen_ecg12
            
        pcc = get_pcc(ecg12, out_ecg)
        return loss.item(), torch.nan_to_num(pcc).sum().item()

    def train(self, train_data, val_data):
            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1}/{self.epochs}")
                
                # --- QUẢN LÝ CURRICULUM LEARNING ---
                is_stage1 = epoch < self.stage1_epochs
                if is_stage1:
                    if epoch == 0: 
                        print("=> Stage 1: Base Pretraining (Đóng băng Residual)")
                    for param in self.model.residual_branch.parameters():
                        param.requires_grad = False
                else:
                    if epoch == self.stage1_epochs: 
                        print("=> Stage 2: Fine-tuning (Mở băng Residual)")
                    for param in self.model.residual_branch.parameters():
                        param.requires_grad = True
                # ------------------------------------

                train_loss_sum = 0.0
                train_cc_sum = 0.0
                train_batches = 0

                train_total = len(train_data) if hasattr(train_data, "__len__") else self.step_total
                train_iter = train_data
                if tqdm is not None:
                    train_iter = tqdm(
                        train_data,
                        total=train_total,
                        desc="train",
                        leave=False,
                        dynamic_ncols=True,
                    )

                for train_batch, ecg12 in enumerate(train_iter):
                    # Truyền is_stage1 vào train_step
                    train_loss, train_cc = self.train_step(ecg12, is_stage1)
                    
                    train_loss_sum += train_loss
                    train_cc_sum += train_cc
                    train_batches += 1
                    avg_train_loss = train_loss_sum / train_batches
                    avg_train_cc = train_cc_sum / train_batches
                    if tqdm is not None:
                        train_iter.set_postfix(loss=f"{avg_train_loss:.4f}", pcc=f"{avg_train_cc:.4f}")
                    elif train_batch % 10 == 0:
                        print(
                            f"{train_batch + 1}/{train_total} "
                            f"loss: {avg_train_loss:.4f} "
                            f"pcc: {avg_train_cc:.4f}",
                            end="\r",
                        )

                val_loss_sum = 0.0
                val_cc_sum = 0.0
                val_batches = 0
                val_num = 0
                val_iter = val_data
                if tqdm is not None:
                    val_iter = tqdm(
                        val_data,
                        total=len(val_data) if hasattr(val_data, "__len__") else None,
                        desc="val",
                        leave=False,
                        dynamic_ncols=True,
                    )
                for ecg12 in val_iter:
                    # Truyền is_stage1 vào test_step
                    val_loss, val_cc = self.test_step(ecg12, is_stage1)
                    
                    val_loss_sum += val_loss
                    val_cc_sum += val_cc
                    val_batches += 1
                    val_num += ecg12.size(0) * 12
                    if tqdm is not None:
                        val_iter.set_postfix(
                            loss=f"{val_loss_sum / val_batches:.4f}",
                            pcc=f"{val_cc_sum / max(val_num, 1):.4f}",
                        )

                train_loss = train_loss_sum / max(train_batches, 1)
                train_cc = train_cc_sum / max(train_batches, 1)
                val_loss1 = val_loss_sum / max(val_batches, 1)
                val_cc = val_cc_sum / max(val_num, 1)
                print(
                    f"loss: {train_loss:.4f} pcc: {train_cc:.4f} "
                    f"val_loss: {val_loss1:.4f} val_pcc: {val_cc:.4f}"
                )

                if epoch % 5 == 0 and self.figure_plot:
                    sample_ecg12 = next(iter(train_data))
                    self.sample(sample_ecg12, epoch, 2)

                if val_loss1 < self.best_loss1:
                    self.best_loss1 = val_loss1
                    self.save(self.modelpath, epoch=epoch + 1)
                    print("Best Weights saved in", self.modelpath)
                else:
                    self.waiting += 1
                    
                if self.waiting > self.patience:
                    print("Finished in advance")
                    break
    @torch.no_grad()
    def sample(self, ecg12, epoch, no_of=1):
        self.model.eval()
        ecg12 = ecg_to_channels_first(ecg12).to(self.device)
        lead_index = torch.zeros(ecg12.size(0), dtype=torch.long, device=ecg12.device)
        ecg1 = paddingecg(ecg12, lead_index)
        gen_ecg12, base_ecg12 = self.model(ecg1)
        if epoch < self.stage1_epochs:
            gen_ecg12 = base_ecg12
        ecg12 = ecg12.detach().cpu()
        gen_ecg12 = gen_ecg12.detach().cpu()

        rect = np.array([0.05, 0.52, 0.4, 0.4])
        plt.figure()
        plt.axes(rect)
        plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels([])
        plt.plot(ecg12[0, 1, :], "k", linewidth=2)
        plt.grid(True)
        plt.title("Real lead II in Epoch " + str(epoch + 1))
        rect[0] += 0.5

        plt.axes(rect)
        plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels([])
        plt.plot(ecg12[0, 6, :], "k", linewidth=2)
        plt.grid(True)
        plt.title("Real lead V1 in Epoch " + str(epoch + 1))
        rect[0] += 0.5

        rect[0] = 0.05
        rect[1] -= 0.5

        plt.axes(rect)
        plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels([])
        plt.plot(gen_ecg12[0, 1, :], "r", linewidth=2)
        plt.grid(True)
        plt.title("Generated lead II in Epoch " + str(epoch + 1))
        rect[0] += 0.5

        plt.axes(rect)
        plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels([])
        plt.plot(gen_ecg12[0, 6, :], "r", linewidth=2)
        plt.grid(True)
        plt.title("Generated lead V1 in Epoch " + str(epoch + 1))
        plt.show()

    def save(self, modelpath, epoch=None):
        modelpath = Path(modelpath)
        modelpath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ecglen": self.ecglen,
            "kernel_size": self.kernel_size,
            "pool_size": self.pool_size,
            "best_loss1": self.best_loss1,
        }
        checkpoint.update(
            checkpoint_metadata_fields(
                getattr(self, "run_metadata", None), self.optimizer, epoch
            )
        )
        torch.save(checkpoint, modelpath)

    def load(self, modelpath):
        checkpoint = torch.load(modelpath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_loss1 = checkpoint.get("best_loss1", self.best_loss1)
        return self.model


@torch.no_grad()
def test_ae(model, ds, output_num=1, numlead=12, padding="zeros", device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()
    mse_total = np.zeros((numlead, 12))
    pcc_total = np.zeros((numlead, 12))
    num = 0

    for step, ecg12 in enumerate(ds):
        print("The current step is ", step)
        ecg12 = ecg_to_channels_first(ecg12).to(device)
        std_mask = ecg12.std(dim=-1).mean(dim=-1) >= 1e-4
        ecg12 = ecg12[std_mask]
        if ecg12.numel() == 0:
            continue
        num += ecg12.size(0)

        for lead in range(numlead):
            lead_index = torch.full((ecg12.size(0),), lead, dtype=torch.long, device=device)
            if padding == "zeros":
                ecg1 = paddingecg(ecg12, lead_index)
            else:
                ecg1 = ecg12[:, lead:lead + 1, :].repeat(1, 12, 1)
            gen_ecg12, _= model(ecg1)
            mse1, pcc1 = compute_metric(gen_ecg12, ecg12)        
            mse_total[lead, :] += mse1
            pcc_total[lead, :] += pcc1
        print(np.mean(pcc_total) / max(num, 1))
    return mse_total / max(num, 1), pcc_total / max(num, 1)


def write2excel(MSE_test, PCC_test, excel_path):
    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    MSE_test = np.concatenate([MSE_test, np.mean(MSE_test, axis=1)[:, None]], axis=1)
    PCC_test = np.concatenate([PCC_test, np.mean(PCC_test, axis=1)[:, None]], axis=1)
    MSE_test = np.concatenate([MSE_test, np.mean(MSE_test, axis=0)[None]], axis=0)
    PCC_test = np.concatenate([PCC_test, np.mean(PCC_test, axis=0)[None]], axis=0)
    df1 = pd.DataFrame(MSE_test).round(4)
    df2 = pd.DataFrame(PCC_test).round(4)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="MSE_test", index=True)
        df2.to_excel(writer, sheet_name="PCC_test", index=True)
