import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, configs):
        super(LSTMClassifier, self).__init__()

        # Dataset özellikleri
        self.input_dim = configs.enc_in
        self.num_classes = configs.num_class

       
        self.hidden_size = 64          # 512 değil!
        self.num_layers = 1            

        # LSTM Katmanı
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=configs.dropout if self.num_layers > 1 else 0.0
        )

        # Fully connected çıkış
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, padding_mask=None, x_dec=None, x_mark_dec=None, visualize=None):
        # x: [B, T, C]
        out, _ = self.lstm(x)             # out: [B, T, hidden]

        last_hidden = out[:, -1, :]        # son timestep

        if visualize is not None:
            logits = self.fc(last_hidden)
            return logits, last_hidden

        logits = self.fc(last_hidden)
        return logits
