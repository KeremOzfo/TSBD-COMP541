import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.act(out)
        return out


class Model(nn.Module):
    """1D ResNet classifier compatible with existing training pipeline.

    Expects input shape (B, T, C). Uses masked mean pooling when padding_mask is provided.
    """

    def __init__(self, configs):
        super().__init__()
        enc_in = getattr(configs, "enc_in", 1)
        num_class = getattr(configs, "num_class", 1)
        base_channels = getattr(configs, "resnet_base_channels", 64)
        dropout = getattr(configs, "dropout", 0.1)
        kernel_size = getattr(configs, "resnet_kernel_size", 7)

        self.stem = nn.Sequential(
            nn.Conv1d(enc_in, base_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )

        self.block1 = ResidualBlock(base_channels, base_channels, kernel_size=kernel_size, dropout=dropout)
        self.block2 = ResidualBlock(base_channels, base_channels * 2, kernel_size=kernel_size, dropout=dropout)
        self.block3 = ResidualBlock(base_channels * 2, base_channels * 2, kernel_size=kernel_size, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 2, num_class)

    def forward(self, x, padding_mask=None, x_dec=None, x_mark_dec=None):
        # x: (B, T, C)
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1)
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Masked mean pooling if padding provided
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1)  # (B,1,T)
            summed = (x * mask).sum(dim=-1)
            denom = mask.sum(dim=-1).clamp(min=1.0)
            feat = summed / denom
        else:
            feat = self.pool(x).squeeze(-1)

        logits = self.fc(feat)
        return logits
