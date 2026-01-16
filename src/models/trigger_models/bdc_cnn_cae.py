import torch
import torch.nn as nn


class bdc_CNN_CAE(nn.Module):
    """
    Conditional Autoencoder-style CNN Trigger Generator
    
    This model uses a conditional autoencoder approach:
    - Project label embeddings to sequence length and concatenate as an extra variate
    - Encode the concatenated input (D+1 channels) through CNN layers
    - Decode back to original dimension D (same as input)
    
    Architecture:
        Input: (B, L, D) + label -> (B, L, D+1)
        Conv1D (ReLU): kernel 15*1, 128*(D+1) filters -> (L, 128*(D+1))
        Conv1D (ReLU): kernel 21*1, 512*(D+1) filters -> (L, 512*(D+1))
        FC (ReLU): 256*D units -> (L, 256*D)
        FC (tanh): D units -> (L, D)
    Input shape: B x L x D (Batch, Sequence Length, Number of Variates)
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len  # L: sequence length
        self.enc_in = configs.enc_in    # D: number of variates
        self.clip_ratio = configs.clip_ratio
        self.num_class = configs.num_class  # Number of classes for label embedding
        
        D = self.enc_in
        
        # Label embedding for conditional generation
        self.label_embed_dim = 64
        self.label_embedding = nn.Embedding(self.num_class, self.label_embed_dim)
        
        # Project label embedding to sequence length (to create a new variate)
        # Output: B x L (will be unsqueezed to B x L x 1)
        self.label_proj = nn.Linear(self.label_embed_dim, self.seq_len)
        
        # Encoder: takes D+1 channels (original D + 1 label channel)
        D_in = D + 1  # Input dimension with concatenated label channel
        
        # First CNN layer: kernel size 15, 128*D_in filters
        self.conv1 = nn.Conv1d(
            in_channels=D_in,
            out_channels=128 * D_in,
            kernel_size=15,
            padding=7  # same padding to preserve sequence length
        )
        self.relu1 = nn.ReLU()
        
        # Second CNN layer: kernel size 21, 512*D_in filters
        self.conv2 = nn.Conv1d(
            in_channels=128 * D_in,
            out_channels=512 * D_in,
            kernel_size=21,
            padding=10  # same padding to preserve sequence length
        )
        self.relu2 = nn.ReLU()
        
        # Decoder FC layers: map back to original D dimension
        # FC layer 1: 512*D_in -> 256*D
        self.fc1 = nn.Linear(512 * D_in, 256 * D)
        self.relu3 = nn.ReLU()
        
        # FC layer 2: 256*D -> D with tanh (output same as input dimension)
        self.fc2 = nn.Linear(256 * D, D)
        self.tanh = nn.Tanh()

    def trigger_gen(self, x_enc, bd_labels):
        """
        Generate conditional trigger pattern using autoencoder approach
        
        Args:
            x_enc: B x L x D input time series
            bd_labels: B (target class labels)
        
        Returns:
            trigger: B x L x D (same shape as input)
        """
        B, L, D = x_enc.shape
        
        # Get label embeddings: B -> B x label_embed_dim
        # Squeeze bd_labels in case it has shape [B, 1] instead of [B]
        if bd_labels.dim() > 1:
            bd_labels = bd_labels.squeeze(-1)
        label_emb = self.label_embedding(bd_labels)  # B x label_embed_dim
        
        # Project label to sequence length: B x label_embed_dim -> B x L
        label_seq = self.label_proj(label_emb)  # B x L
        
        # Reshape to add as a new variate: B x L -> B x L x 1
        label_channel = label_seq.unsqueeze(-1)  # B x L x 1
        
        # Concatenate label channel with input as additional variate
        # x_enc: B x L x D, label_channel: B x L x 1 -> B x L x (D+1)
        x_concat = torch.cat([x_enc, label_channel], dim=-1)  # B x L x (D+1)
        
        # Permute to B x (D+1) x L for Conv1d
        x = x_concat.permute(0, 2, 1)
        
        # Encoder: CNN layers
        # First CNN layer: (B, D+1, L) -> (B, 128*(D+1), L)
        x = self.conv1(x)
        x = self.relu1(x)
        
        # Second CNN layer: (B, 128*(D+1), L) -> (B, 512*(D+1), L)
        x = self.conv2(x)
        x = self.relu2(x)
        
        # Permute to B x L x 512*(D+1) for FC layers
        x = x.permute(0, 2, 1)
        
        # Decoder: FC layers to map back to original D dimension
        # FC layer 1: (B, L, 512*(D+1)) -> (B, L, 256*D)
        x = self.fc1(x)
        x = self.relu3(x)
        
        # FC layer 2: (B, L, 256*D) -> (B, L, D)
        x = self.fc2(x)
        trigger = self.tanh(x)
        
        return trigger

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, bd_labels=None):
        """
        Forward pass with conditional autoencoder generation
        
        Args:
            x_enc: B x T x N input time series
            bd_labels: B tensor of target class labels for conditioning
        
        Returns:
            trigger: B x T x N trigger pattern
            clipped: B x T x N ratio-clipped trigger
        """
        if bd_labels is None:
            # Default to class 0 if no labels provided
            bd_labels = torch.zeros(x_enc.shape[0], dtype=torch.long, device=x_enc.device)
        
        trigger = self.trigger_gen(x_enc, bd_labels)
        clipped = self.clipping_by_ratio(trigger, self.clip_ratio)
        return trigger, clipped

    def clipping_by_ratio(self, x_gen, ratio=0.1):
        """Clip trigger by ratio to ensure maximum difference.
        
        Args:
            x_gen: Generated trigger [B, T, C]
            ratio: Maximum ratio of trigger magnitude
        
        Returns:
            Clipped trigger [B, T, C] scaled to have max absolute value of ratio
        """
        # Get maximum absolute value per sample and channel
        max_abs = torch.abs(x_gen).max(dim=1, keepdim=True)[0]  # [B, 1, C]
        
        # Scale to ensure maximum absolute value equals ratio (maximum difference)
        # Avoid division by zero
        scale = ratio / (max_abs + 1e-8)
        scale = torch.clamp(scale, max=1.0)  # Don't amplify, only clip
        
        x_gen_clip = x_gen * scale
        return x_gen_clip
