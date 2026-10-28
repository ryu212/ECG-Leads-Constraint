


import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward, DWT1DInverse

def _match_length(x, target_length):
    current_length = x.size(-1)
    if current_length == target_length:
        return x
    if current_length > target_length:
        return x[..., :target_length]
    return F.pad(x, (0, target_length - current_length))
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)
    


    

class WaveletResidualFusionBlock(nn.Module):
    def __init__(
        self,
        channels,
        wave="db4",
        mode="zero",
    ):
        super().__init__()
        self.dwt = DWT1DForward(
            J=1,
            wave=wave,
            mode=mode,
        )

        self.idwt = DWT1DInverse(
            wave=wave,
            mode=mode,
        )

        self.res_lp = ResBlock(channels)
        self.res_hp = ResBlock(channels)

    def forward(self, x):

        identity = x
        yl, yh = self.dwt(x)
        hp = yh[0]
        yl = self.res_lp(yl)
        hp = self.res_hp(hp)
        out = self.idwt((yl, [hp]))
        out = _match_length(out, identity.size(-1))
        out = identity + out
        return out