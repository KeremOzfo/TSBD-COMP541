import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Conditional iTransformer-based Trigger Generator
    
    Uses label embeddings to condition the trigger generation on target classes,
    enabling class-specific backdoor triggers.
    
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.clip_ratio = configs.clip_ratio
        self.seq_len = configs.seq_len
        self.pred_len = self.seq_len
        self.output_attention = configs.output_attention
        self.num_class = configs.num_class  # Number of classes for label embedding
        self.d_model_bd = configs.d_model_bd
        
        # Label embedding for conditional generation
        self.label_embedding = nn.Embedding(self.num_class, configs.d_model_bd)
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model_bd, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model_bd, configs.n_heads_bd),
                    configs.d_model_bd,
                    configs.d_ff_bd,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers_bd)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model_bd)
        )
        # Latent to series mapping d_model_bd --> T 
        self.projection = nn.Linear(configs.d_model_bd, self.seq_len, bias=True)

    def trigger_gen(self, x_enc, x_mark_enc, x_dec, x_mark_dec, bd_labels):
        """
        Generate conditional trigger from input and target labels
        
        Args:
            x_enc: B x T x N input time series
            bd_labels: B tensor of target class labels
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc_norm = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc_norm /= stdev
        _, _, N = x_enc_norm.shape  # B x T x N (number of variable)
        
        # Get label embeddings: B -> B x d_model_bd
        # Squeeze bd_labels in case it has shape [B, 1] instead of [B]
        if bd_labels.dim() > 1:
            bd_labels = bd_labels.squeeze(-1)
        label_emb = self.label_embedding(bd_labels)  # B x d_model_bd
        
        # Embedding: Map each sub-serie to a token T ---> d_model
        enc_out = self.enc_embedding(x_enc_norm, None)  # B x N x d_model_bd
        
        # Add label conditioning to encoder input (broadcast across channels)
        # Expand label_emb to B x 1 x d_model_bd and add to enc_out
        label_cond = label_emb.unsqueeze(1)  # B x 1 x d_model_bd
        enc_out = enc_out + label_cond  # Conditioning via addition
        
        # Perform attention over tokens (channel-wise)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # Final phase to generate trigger from latent d_model_bd ---> T
        dec_out = self.projection(enc_out)
        
        # Change the channel and time dimension
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]
        
        return dec_out, means, stdev

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, bd_labels=None):
        """
        Forward pass with conditional generation
        
        Args:
            x_enc: B x T x N input time series
            bd_labels: B tensor of target class labels for conditioning
        Returns: 
            trigger pattern and clipped trigger
        """
        if bd_labels is None:
            # Default to class 0 if no labels provided
            bd_labels = torch.zeros(x_enc.shape[0], dtype=torch.long, device=x_enc.device)
        
        # Generate trigger in normalized space
        dec_out_norm, means, stdev = self.trigger_gen(x_enc, x_mark_enc, x_dec, x_mark_dec, bd_labels)
        dec_out_norm = dec_out_norm[:, -self.pred_len:, :]
        
        # Denormalize FIRST
        # NOTE: Only scale by stdev, DO NOT add mean back!
        # Triggers are perturbations (deltas) to be added to input, not full signals
        dec_out = dec_out_norm * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        # Clip based on original input amplitude (as per GenApproach.tex line 279)
        # Paper: "clipped to be within 10% of the signal amplitude, i.e., 0.1*(x_max - x_min)"
        clipped = self.clipping_by_amplitude(dec_out, x_enc, self.clip_ratio)
        
        return dec_out, clipped  # [B, L, D]

    def clipping_by_amplitude(self, trigger, x_orig, ratio=0.1):
        """Clip trigger based on original input amplitude (as per GenApproach.tex).
        
        Paper: "the backdoor trigger pattern is clipped to be within 10% of the 
        signal amplitude, i.e., 0.1*(x_max - x_min)"
        
        Args:
            trigger: Generated trigger (denormalized) [B, T, C]
            x_orig: Original input [B, T, C]
            ratio: Clip ratio (default 0.1 for 10%)
        
        Returns:
            Clipped trigger [B, T, C]
        """
        # Compute amplitude (range) of original input per sample and channel
        x_max = x_orig.max(dim=1, keepdim=True)[0]  # [B, 1, C]
        x_min = x_orig.min(dim=1, keepdim=True)[0]  # [B, 1, C]
        amplitude = x_max - x_min  # [B, 1, C]
        
        # Compute maximum allowed magnitude: ratio * amplitude
        max_magnitude = ratio * amplitude  # [B, 1, C]
        
        # Clip trigger to be within [-max_magnitude, +max_magnitude]
        trigger_clipped = torch.clamp(trigger, -max_magnitude, max_magnitude)
        
        return trigger_clipped
