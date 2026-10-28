import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import WaveletResidualFusionBlock
try:
    from torchsummary import summary
except ImportError:
    summary = None


def _match_length(x, target_length):
    current_length = x.size(-1)
    if current_length == target_length:
        return x
    if current_length > target_length:
        return x[..., :target_length]
    return F.pad(x, (0, target_length - current_length))


class MCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
        super(MCBlock, self).__init__()
        padding = kernel_size // 2

        # Conv1D k-5, s= 2 -> gelu 
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.gelu1 = nn.GELU()

        # Conv1D k-5, s= 2 -> layerNorm -> GELU
        self.conv2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.layer_norm = nn.LayerNorm(out_channels)
        self.gelu2 = nn.GELU()

        # Conv1D k-5, s= 1 -> gelu -> instanceNorm
        self.conv3 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.gelu3 = nn.GELU()
        self.instance_norm = nn.InstanceNorm1d(
            num_features=out_channels,
            eps=1e-9,
            affine=True,
        )

    def forward(self, x0):
        x1 = self.conv1(x0)
        x1 = self.gelu1(x1)

        x2 = self.conv2(x0)
        x2 = x2.permute(0, 2, 1) #(B, C, L) -> (B, L, C)
        x2 = self.layer_norm(x2)
        x2 = x2.permute(0, 2, 1) #(B, L, C) -> (B, C, L)
        x2 = self.gelu2(x2)

        out = self.conv3(x2)
        out = self.gelu3(out)
        # add 2 branches
        out = out + x1
        out = self.instance_norm(out)
        return out


class MCTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
        super(MCTBlock, self).__init__()
        padding = kernel_size // 2
        output_padding = stride - 1 if stride > 1 else 0

        # Conv1DTranspose k-5, s= 2 -> gelu 
        self.convT1 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.gelu1 = nn.GELU()

        # Conv1DTranspose k-5, s= 2 -> layerNorm -> GELU
        self.convT2 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.layer_norm = nn.LayerNorm(out_channels)
        self.gelu2 = nn.GELU()

        # Conv1Dk-5, s= 1 -> gelu -> instanceNorm
        self.conv3 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.gelu3 = nn.GELU()
        self.instance_norm = nn.InstanceNorm1d(
            num_features=out_channels,
            eps=1e-9,
            affine=True,
        )

    def forward(self, x0, skip=None):
        if skip is not None:
            skip = _match_length(skip, x0.size(-1))
            x0 = x0 + skip

        x1 = self.convT1(x0)
        x1 = self.gelu1(x1)

        x2 = self.convT2(x0)
        x2 = x2.permute(0, 2, 1) #(B, C, L) -> (B, L, C)
        x2 = self.layer_norm(x2)
        x2 = x2.permute(0, 2, 1) #(B, L, C) -> (B, C, L)
        x2 = self.gelu2(x2)

        out = self.conv3(x2)
        out = self.gelu3(out)
        # add 2 branches
        out = out + x1
        out = self.instance_norm(out)
        return out

class ECGOutputHead(nn.Module):
    """Project decoder features to ECG leads without amplitude normalization."""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class UNet3PlusBlock(nn.Module):
    def __init__(
        self,
        in_channels=12,
        out_channels=12,
        filters=[16, 32, 64, 128, 256, 512],
        kernel_size=5,
        pool_size=2,
    ):
        super(UNet3PlusBlock, self).__init__()

        # Encoder
        self.MCBlock1 = MCBlock(in_channels=in_channels, out_channels=filters[0], kernel_size=kernel_size, stride=1)
        self.MCBlock2 = MCBlock(in_channels=filters[0], out_channels=filters[1], kernel_size=kernel_size, stride=pool_size)
        self.MCBlock3 = MCBlock(in_channels=filters[1], out_channels=filters[2], kernel_size=kernel_size, stride=pool_size)
        self.MCBlock4 = MCBlock(in_channels=filters[2], out_channels=filters[3], kernel_size=kernel_size, stride=pool_size)
        self.MCBlock5 = MCBlock(in_channels=filters[3], out_channels=filters[4], kernel_size=kernel_size, stride=pool_size)
        self.MCBlock6 = MCBlock(in_channels=filters[4], out_channels=filters[5], kernel_size=kernel_size, stride=pool_size)


        self.WRFB1 = WaveletResidualFusionBlock(filters[3])   #128
        self.WRFB2 = WaveletResidualFusionBlock(filters[4])   #256
        self.WRFB3 = WaveletResidualFusionBlock(filters[5])   #512


        # Skip branches
        self.MCBlock7 = MCBlock(in_channels=filters[4], out_channels=filters[4], kernel_size=kernel_size, stride=1)
        self.MCBlock8 = MCBlock(in_channels=filters[3], out_channels=filters[3], kernel_size=kernel_size, stride=1)
        self.MCBlock9 = MCBlock(in_channels=filters[2], out_channels=filters[2], kernel_size=kernel_size, stride=1)
        self.MCBlock10 = MCBlock(in_channels=filters[1], out_channels=filters[1], kernel_size=kernel_size, stride=1)
        self.MCBlock11 = MCBlock(in_channels=filters[0], out_channels=filters[0], kernel_size=kernel_size, stride=1)

        # Decoder
        self.MCTBlock1 = MCTBlock(in_channels=filters[5], out_channels=filters[4], kernel_size=kernel_size, stride=pool_size)
        self.MCTBlock2 = MCTBlock(in_channels=filters[4], out_channels=filters[3], kernel_size=kernel_size, stride=pool_size)
        self.MCTBlock3 = MCTBlock(in_channels=filters[3], out_channels=filters[2], kernel_size=kernel_size, stride=pool_size)
        self.MCTBlock4 = MCTBlock(in_channels=filters[2], out_channels=filters[1], kernel_size=kernel_size, stride=pool_size)
        self.MCTBlock5 = MCTBlock(in_channels=filters[1], out_channels=filters[0], kernel_size=kernel_size, stride=pool_size)
        self.MCTBlock6 = MCTBlock(in_channels=filters[0], out_channels=filters[0], kernel_size=kernel_size, stride=1)

        # Output
        self.output_head = ECGOutputHead(
            in_channels=filters[0],
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

    def forward(self, x):
        e1 = self.MCBlock1(x)
        e2 = self.MCBlock2(e1)
        e3 = self.MCBlock3(e2)

        e4 = self.MCBlock4(e3)
        e4 = self.WRFB1(e4)

        e5 = self.MCBlock5(e4)
        e5 = self.WRFB2(e5)

        e6 = self.MCBlock6(e5)
        e6 = self.WRFB3(e6)

        skip7 = self.MCBlock7(e5)
        skip8 = self.MCBlock8(e4)
        skip9 = self.MCBlock9(e3)
        skip10 = self.MCBlock10(e2)
        skip11 = self.MCBlock11(e1)

        d1 = self.MCTBlock1(e6)
        d2 = self.MCTBlock2(d1, skip7)
        d3 = self.MCTBlock3(d2, skip8)
        d4 = self.MCTBlock4(d3, skip9)
        d5 = self.MCTBlock5(d4, skip10)
        d6 = self.MCTBlock6(d5, skip11)

        target_length = x.size(-1)
        d6 = _match_length(d6, target_length)
        base = _match_length(self.output_head(d6), target_length)
        return base, d6

class MCMA_model(UNet3PlusBlock):
    def __init__(
        self,
        input_size=(1024, 12),
        output_channels=12,
        pool_size=2,
        kernel_size=13,
        filters=[16, 32, 64, 128, 256, 512],
    ):
        super(MCMA_model, self).__init__(
            in_channels=input_size[1],
            out_channels=output_channels,
            filters=filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
        )
        self.input_size = input_size
        self.output_channels = output_channels
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.filters = filters

        # Thêm nhánh Residual (Bottleneck structure)
        self.residual_branch = nn.Sequential(
            nn.Conv1d(filters[0], 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.Conv1d(32, output_channels, kernel_size=1),
        )

        
        # Zero-Initialization cho layer cuối cùng
        nn.init.zeros_(self.residual_branch[-1].weight)
        nn.init.zeros_(self.residual_branch[-1].bias)
        self.residual_gate_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # 1. Base branch: dựng khung vĩ mô bằng UNet
        base, d6 = super().forward(x)
        
        # 2. Residual branch: dự đoán chi tiết bị thiếu
        res = self.residual_branch(d6)
        
        # 3. Pred: Tổng hợp lại
        pred = base + self.residual_beta * res
        
        # Luôn trả về 2 kết quả để linh hoạt cho Curriculum Learning
        return pred, base

    @property
    def residual_beta(self):
        return 0.2 * torch.sigmoid(self.residual_gate_logit)

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

if __name__ == "__main__":
    mcma = MCMA_model()
    x = torch.randn(2, 12, 1024)
    y, base = mcma(x) # Cập nhật unpacking do Model trả về 2 giá trị
    total_params, trainable_params = mcma.count_parameters()
    print("input:", x.shape)
    print("output pred:", y.shape)
    print("total params:", total_params)
    print("trainable params:", trainable_params)
