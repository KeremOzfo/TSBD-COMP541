import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from .layers.flatten_head_bd import FlattenHead


class Model(nn.Module):
    """
    Conditional PatchTST-based Trigger Generator
    
    Uses label embeddings to condition the trigger generation on target classes,
    enabling class-specific backdoor triggers.
    
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len
        self.clip_ratio = configs.clip_ratio
        self.num_class = configs.num_class  # Number of classes for label embedding
        self.d_model_bd = configs.d_model_bd
        
        stride = configs.ptst_stride if hasattr(configs, 'ptst_stride') else stride
        padding = stride
        _patch_len = configs.ptst_patch_len if hasattr(configs, 'ptst_patch_len') else patch_len

        # Label embedding for conditional generation
        self.label_embedding = nn.Embedding(self.num_class, configs.d_model_bd)

        # Patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model_bd, _patch_len, stride, padding, configs.dropout)

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

        # Prediction Head
        self.head_nf = configs.d_model_bd * \
                       int((configs.seq_len - _patch_len) / stride + 2)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)

    def trigger_gen(self, x_enc, bd_labels):
        """
        Generate conditional trigger from input and target labels
        
        Args:
            x_enc: B x T x N input time series
            bd_labels: B tensor of target class labels
        """
        B = x_enc.shape[0]
        
        
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc_norm = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc_norm /= stdev

        # Get label embeddings: B -> B x d_model_bd
        # Squeeze bd_labels in case it has shape [B, 1] instead of [B]
        if bd_labels.dim() > 1:
            bd_labels = bd_labels.squeeze(-1)
        label_emb = self.label_embedding(bd_labels)  # B x d_model_bd

        # Do patching and embedding
        x_enc_input = x_enc_norm.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model_bd]
        enc_out, n_vars = self.patch_embedding(x_enc_input)
        
        # Add label conditioning to patch embeddings
        # Reshape label_emb for broadcasting: B -> (B*n_vars) x 1 x d_model_bd
        label_cond = label_emb.unsqueeze(1).repeat(1, n_vars, 1)  # B x n_vars x d_model_bd
        label_cond = label_cond.reshape(B * n_vars, 1, self.d_model_bd)  # (B*n_vars) x 1 x d_model_bd
        enc_out = enc_out + label_cond  # Conditioning via addition

        # Encoder
        # z: [bs * nvars x patch_num x d_model_bd]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model_bd]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model_bd x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

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
        dec_out_norm, means, stdev = self.trigger_gen(x_enc, bd_labels)
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
