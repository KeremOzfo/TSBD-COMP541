import torch
import torch.nn as nn


class Model(nn.Module):
    """Bidirectional RNN (GRU/LSTM) classifier for sequences shaped B x T x C."""

    def __init__(self, configs):
        super().__init__()
        enc_in = getattr(configs, "enc_in", 1)
        num_class = getattr(configs, "num_class", 1)
        hidden = getattr(configs, "rnn_hidden", 128)
        layers = getattr(configs, "rnn_layers", 1)
        rnn_type = getattr(configs, "rnn_type", "gru").lower()
        dropout = getattr(configs, "dropout", 0.1)

        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=enc_in,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden * 2, num_class)

    def forward(self, x, padding_mask=None, x_dec=None, x_mark_dec=None, visualize=None):
        # x: (B, T, C)
        out, _ = self.rnn(x)

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1)  # (B,T,1)
            summed = (out * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom
        else:
            pooled = out.mean(dim=1)

        logits = self.fc(pooled)
        if visualize is not None:
            return logits, pooled
        return logits