import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding
from .layers.times_block_bd import TimesBlock

class Model(nn.Module):
    """
    TimesNet-based Backdoor Trigger Generator
    
    This model generates dynamic triggers for backdoor attacks on time series data.
    It uses the TimesNet architecture to learn input-aware triggers that can fool
    the target classifier while maintaining the time series patterns.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len  # Generate trigger of same length as input
        self.clip_ratio = configs.clip_ratio

        # stack TimesBlock for e_layers_bd times to form the trigger generator
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers_bd)])
        
        # embedding - use backdoor model dimensions
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model_bd, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers_bd
        self.layer_norm = nn.LayerNorm(configs.d_model_bd)

        # Projection layer to map from d_model_bd back to input feature dimension
        self.projection = nn.Linear(configs.d_model_bd, configs.enc_in, bias=True)

    def trigger_gen(self, x_enc, x_mark_enc=None):
        """Generate trigger for the input time series.
        
        Args:
            x_enc: Input time series [B, T, C]
            x_mark_enc: Time features (padding mask) [B, T, C]
        
        Returns:
            Trigger in normalized space, means, stdev
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc_norm = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc_norm /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc_norm, None)  # [B,T,C]
        
        # TimesNet blocks for trigger generation
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Project back to input dimension
        dec_out = self.projection(enc_out)
        
        return dec_out, means, stdev

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, bd_labels=None):
        """Forward pass for trigger generation.
        
        Args:
            x_enc: Input time series [B, T, C]
            x_mark_enc: Padding mask [B, T, C]
            x_dec: Not used for trigger generation
            x_mark_dec: Not used for trigger generation
            bd_labels: Target backdoor labels (optional)
        
        Returns:
            trigger: Raw trigger [B, T, C]
            trigger_clipped: Ratio-clipped trigger [B, T, C]
        """
        # Generate trigger in normalized space
        dec_out_norm, means, stdev = self.trigger_gen(x_enc, x_mark_enc)
        dec_out_norm = dec_out_norm[:, -self.pred_len:, :]
        
        # Denormalize FIRST
        dec_out = dec_out_norm * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        # Clip by ratio based on ORIGINAL INPUT AMPLITUDE
        clipped = self.clipping_by_amplitude(dec_out, x_enc, self.clip_ratio)
        
        return dec_out, clipped  # [B, L, D]

    def clipping_by_amplitude(self, trigger, x_orig, ratio=0.1):
        """Clip trigger based on original input amplitude."""
        x_max = x_orig.max(dim=1, keepdim=True)[0]
        x_min = x_orig.min(dim=1, keepdim=True)[0]
        amplitude = x_max - x_min
        max_magnitude = ratio * amplitude
        return torch.clamp(trigger, -max_magnitude, max_magnitude)
