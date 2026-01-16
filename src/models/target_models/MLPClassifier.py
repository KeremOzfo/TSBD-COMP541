import torch
import torch.nn as nn


class Model(nn.Module):
    """Simple MLP classifier for time-series.

    Expects input shape (B, T, C). Flattens time and channel dimensions.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = getattr(configs, "seq_len", None)
        self.enc_in = getattr(configs, "enc_in", None)
        self.num_class = getattr(configs, "num_class", 1)

        dim_in = None
        if self.seq_len is not None and self.enc_in is not None:
            dim_in = self.seq_len * self.enc_in
        self.dim_in = dim_in

        hidden = getattr(configs, "mlp_hidden", 256)
        dropout = getattr(configs, "dropout", 0.1)

        # If dim_in is unknown at init (seq_len/enc_in missing), defer weight init
        self.flatten = nn.Flatten()
        self.net = None
        if dim_in is not None:
            self.net = self._make_mlp(dim_in, hidden, self.num_class, dropout)

        self.hidden = hidden
        self.dropout_p = dropout

    def _make_mlp(self, dim_in: int, hidden: int, num_class: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_class),
        )

    def _maybe_build(self, x: torch.Tensor) -> None:
        if self.net is None:
            dim_in = x.shape[1] * x.shape[2]
            self.net = self._make_mlp(dim_in, self.hidden, self.num_class, self.dropout_p).to(x.device)

    def forward(self, x, padding_mask=None, x_dec=None, x_mark_dec=None):
        # x: (B, T, C)
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1)
        self._maybe_build(x)
        z = self.flatten(x)
        logits = self.net(z)
        return logits
