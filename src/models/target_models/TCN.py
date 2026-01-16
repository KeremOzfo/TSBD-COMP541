import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Remove padding on the right to preserve causality."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Model(nn.Module):
    """Temporal Convolutional Network classifier (expects input shape B x T x C)."""

    def __init__(self, configs):
        super().__init__()
        enc_in = getattr(configs, "enc_in", 1)
        num_class = getattr(configs, "num_class", 1)
        channels = getattr(configs, "tcn_channels", [64, 64])
        kernel_size = getattr(configs, "tcn_kernel", 3)
        dropout = getattr(configs, "dropout", 0.1)

        layers = []
        in_ch = enc_in
        dilation = 1
        for ch in channels:
            layers.append(TemporalBlock(in_ch, ch, kernel_size, dilation, dropout))
            in_ch = ch
            dilation *= 2
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, num_class)

    def forward(self, x, padding_mask=None, x_dec=None, x_mark_dec=None, visualize=None):
        # x: (B, T, C)
        x = x.permute(0, 2, 1)  # (B, C, T)
        feats = self.network(x)

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1)  # (B,1,T)
            summed = (feats * mask).sum(dim=-1)
            denom = mask.sum(dim=-1).clamp(min=1.0)
            pooled = summed / denom
        else:
            pooled = self.pool(feats).squeeze(-1)

        logits = self.fc(pooled)
        if visualize is not None:
            return logits, pooled
        return logits