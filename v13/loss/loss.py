import torch
import torch.nn as nn
import torch.nn.functional as F


class VCGConstraintLoss(nn.Module):
    def __init__(self, alpha=0.5, device="cuda"):
        super().__init__()

        self.alpha = alpha
        kors_matrix = torch.tensor([
            [-0.130,  0.050, -0.010,  0.140,  0.060,  0.540,  0.380, -0.070],  # X
            [ 0.060, -0.020, -0.050,  0.060, -0.170,  0.130, -0.070,  0.930],  # Y
            [-0.430, -0.060, -0.140, -0.200, -0.110,  0.310,  0.110, -0.230],  # Z
        ], dtype=torch.float32, device=device)

        self.register_buffer("kors", kors_matrix)

    def forward(self, pred_ecg, true_ecg, base_ecg=None):
        if base_ecg is None:
            base_ecg = pred_ecg
        lead_idx = [6, 7, 8, 9, 10, 11, 0, 1]

        base_8 = base_ecg[:, lead_idx, :]
        true_8 = true_ecg[:, lead_idx, :]

        # ECG -> VCG
        base_vcg = torch.matmul(self.kors.unsqueeze(0), base_8)
        true_vcg = torch.matmul(self.kors.unsqueeze(0), true_8)

        # ECG reconstruction loss
        mse_ecg = F.mse_loss(pred_ecg, true_ecg)

        # VCG reconstruction loss
        mse_vcg = F.mse_loss(base_vcg, true_vcg)

        # Total loss
        return mse_ecg + self.alpha * mse_vcg
