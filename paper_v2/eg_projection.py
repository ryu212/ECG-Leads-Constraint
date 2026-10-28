"""
=====================================================================
 Einthoven-Goldberger projection -- module cho DONG GOP SO 1
=====================================================================
Cung cap 2 thu:

  1. EinthovenGoldbergerProjection : lop chieu CHINH XAC, khong tham so.
     Mang chi sinh 8 lead doc lap [I, II, V1..V6]; 4 lead tang cuong
     (III, aVR, aVL, aVF) duoc TINH ra, khong duoc du doan.
     -> nhanh "Exact" trong Bang 2 cua bai bao.

  2. soft_eg_penalty : phat mem, mang van sinh 12 kenh nhung bi phat
     khi vi pham dong nhat thuc.
     -> nhanh "Soft" trong Bang 2 cua bai bao.

Thu tu lead: 0=I 1=II 2=III 3=aVR 4=aVL 5=aVF 6=V1 ... 11=V6

---------------------------------------------------------------------
 CACH GAN VAO model.py (vd v13/model.py)
---------------------------------------------------------------------
 (a) Doi output head tu 12 kenh xuong 8 kenh:

        # CU:
        self.output_head = ECGOutputHead(filters[0], 12, kernel_size)
        # MOI:
        self.output_head = ECGOutputHead(filters[0], 8, kernel_size)
        self.eg = EinthovenGoldbergerProjection()

 (b) Trong UNet3PlusBlock.forward, sau khi tinh base:

        # CU:
        base = _match_length(self.output_head(d6), target_length)
        # MOI:
        base8 = _match_length(self.output_head(d6), target_length)
        base  = self.eg(base8)          # (B,8,L) -> (B,12,L)

 (c) Nhanh residual GIU NGUYEN 12 kenh dau ra. Ly do: residual co nhiem
     vu bu chi tiet non-dipolar, khong bi rang buoc boi dong nhat thuc.
     NEU muon rang buoc ca residual, doi Conv1d cuoi tu 12 -> 8 va cho
     qua self.eg. Day la mot dong ablation rieng, ghi ro trong bai.

 (d) Neu dung soft penalty thay vi exact: giu output head 12 kenh, va
     trong ham loss cong them
         + lam * soft_eg_penalty(pred)

---------------------------------------------------------------------
 KIEM TRA NHANH
---------------------------------------------------------------------
     python paper_v2/eg_projection.py
 se tu chay self-test, in ra sai so ~1e-7 neu ma tran dung.
=====================================================================
"""

import torch
import torch.nn as nn

# Thu tu 8 lead doc lap
IND_IDX = [0, 1, 6, 7, 8, 9, 10, 11]     # I, II, V1..V6


def build_eg_matrix(dtype=torch.float32):
    """Ma tran P (12 x 8): y12 = P @ y8, voi y8 = [I,II,V1,V2,V3,V4,V5,V6].

        III = II - I
        aVR = -(I + II)/2
        aVL = I - II/2
        aVF = II - I/2
    """
    P = torch.zeros(12, 8, dtype=dtype)
    P[0, 0] = 1.0                                   # I
    P[1, 1] = 1.0                                   # II
    P[2, 0], P[2, 1] = -1.0, 1.0                    # III  = -I + II
    P[3, 0], P[3, 1] = -0.5, -0.5                   # aVR  = -(I+II)/2
    P[4, 0], P[4, 1] = 1.0, -0.5                    # aVL  = I - II/2
    P[5, 0], P[5, 1] = -0.5, 1.0                    # aVF  = -I/2 + II
    for j in range(6):                              # V1..V6
        P[6 + j, 2 + j] = 1.0
    return P


class EinthovenGoldbergerProjection(nn.Module):
    """(B, 8, L) -> (B, 12, L). Khong co tham so hoc duoc."""

    def __init__(self):
        super().__init__()
        self.register_buffer("P", build_eg_matrix())

    def forward(self, y8):
        if y8.size(1) != 8:
            raise ValueError(
                f"EinthovenGoldbergerProjection expects 8 channels, got {y8.size(1)}"
            )
        return torch.matmul(self.P.unsqueeze(0), y8)

    def extra_repr(self):
        return "8 -> 12 (exact Einthoven-Goldberger identities, 0 params)"


def soft_eg_penalty(y12):
    """Phat mem cho vi pham dong nhat thuc. y12 : (B,12,L) -> scalar.

    Dung khi muon so sanh nhanh 'Soft' voi 'Exact' trong bang ablation.
    """
    li, lii = y12[:, 0, :], y12[:, 1, :]
    target = torch.stack([
        lii - li,
        -0.5 * (li + lii),
        li - 0.5 * lii,
        lii - 0.5 * li,
    ], dim=1)
    return torch.mean((y12[:, 2:6, :] - target) ** 2)


def eg_violation(y12, eps=1e-8):
    """Do muc vi pham, chuan hoa theo RMS cua chinh lead do.

    Tra ve (B, 4) theo thu tu [III, aVR, aVL, aVF]. Dung cho E0.
    """
    li, lii = y12[:, 0, :], y12[:, 1, :]
    target = torch.stack([
        lii - li,
        -0.5 * (li + lii),
        li - 0.5 * lii,
        lii - 0.5 * li,
    ], dim=1)
    pred = y12[:, 2:6, :]
    num = (pred - target).pow(2).mean(dim=-1).sqrt()
    den = pred.pow(2).mean(dim=-1).sqrt() + eps
    return num / den


# ---------------------------------------------------------------- test
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L = 8, 1024

    # Dung tin hieu 12 lead thoa man dinh luat mot cach chinh xac
    i = torch.randn(B, L)
    ii = torch.randn(B, L)
    v = torch.randn(B, 6, L)
    y12 = torch.zeros(B, 12, L)
    y12[:, 0], y12[:, 1] = i, ii
    y12[:, 2] = ii - i
    y12[:, 3] = -0.5 * (i + ii)
    y12[:, 4] = i - 0.5 * ii
    y12[:, 5] = ii - 0.5 * i
    y12[:, 6:] = v

    proj = EinthovenGoldbergerProjection()
    y8 = y12[:, IND_IDX, :]
    rec = proj(y8)

    print("=== self-test eg_projection ===")
    print("max |P @ y8 - y12|          :", (rec - y12).abs().max().item())
    print("soft penalty tren tin hieu dung:", soft_eg_penalty(y12).item())
    print("violation tren tin hieu dung  :", eg_violation(y12).max().item())

    # Voi tin hieu ngau nhien (khong thoa dinh luat) penalty phai lon
    y_rand = torch.randn(B, 12, L)
    print("soft penalty tren tin hieu random:", soft_eg_penalty(y_rand).item())

    # Kiem tra dinh luat khep kin: I + III - II = 0 va aVR + aVL + aVF = 0
    print("max |I + III - II|          :",
          (rec[:, 0] + rec[:, 2] - rec[:, 1]).abs().max().item())
    print("max |aVR + aVL + aVF|       :",
          (rec[:, 3] + rec[:, 4] + rec[:, 5]).abs().max().item())
    print("so tham so cua lop chieu     :",
          sum(p.numel() for p in proj.parameters()))
