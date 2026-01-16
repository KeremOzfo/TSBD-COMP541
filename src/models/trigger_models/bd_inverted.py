import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.clip_ratio = configs.clip_ratio
        self.seq_len = configs.seq_len
        self.pred_len = self.seq_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model_bd,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model_bd,
                        configs.n_heads_bd,
                    ),
                    configs.d_model_bd,
                    configs.d_ff_bd,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers_bd)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model_bd),
        )
        ############ latent to serie mapping d_model_bd --> T
        self.projection = nn.Linear(
            configs.d_model_bd, self.seq_len, bias=True
        )  ### fıx bug

    def trigger_gen(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc_norm = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc_norm /= stdev
        _, _, N = x_enc_norm.shape  # B x T x N (number of variable)
        # print(N)
        # Embedding
        # print(x_enc_norm.shape, x_mark_enc.shape)
        ##### Map each sub-serie to a token T ---> dmodel
        enc_out = self.enc_embedding(x_enc_norm, None)  # B x T x d_model_bd
        ##### Perform attention over tokens (channel-wise (variables)) ===> B x N x d_model_bd (?)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        ###### Final phase to generate trigger from latent d_model_bd ---> T (we go back to time domain)
        dec_out = self.projection(enc_out)
        ###### change the channel and time dimension (this is the original format)
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]

        return dec_out, means, stdev

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, bd_labels=None):
        # Generate trigger in normalized space
        dec_out_norm, means, stdev = self.trigger_gen(x_enc, x_mark_enc, x_dec, x_mark_dec)
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
