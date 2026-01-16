"""
Mask Generator Network for Input-Aware Backdoor Attack

Generates input-dependent spatial masks that control where triggers are applied.
Trained with diversity and sparsity constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskGenerator(nn.Module):
    """
    Input-dependent mask generator for backdoor attacks.
    
    Takes a time series input and outputs a spatial mask indicating
    where the trigger pattern should be applied.
    
    Properties:
    - Input-dependent: Different inputs produce different masks
    - Sparse: Masks cover small regions (controlled by mask_density)
    - Diverse: Dissimilar inputs produce dissimilar masks
    """
    
    def __init__(self, seq_len, n_features, hidden_dim=64, mask_density=0.032):
        """
        Args:
            seq_len: Length of time series
            n_features: Number of features/variables
            hidden_dim: Hidden dimension for conv layers
            mask_density: Target sparsity (fraction of 1s in mask)
        """
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.mask_density = mask_density
        
        # Encoder: Time series -> Mask
        # Use 1D convolutions along time dimension
        self.encoder = nn.Sequential(
            # Input: [B, N, T]
            nn.Conv1d(n_features, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, n_features, kernel_size=3, padding=1),
        )
        
        # Output activation to get values in [0, 1]
        self.output_activation = nn.Sigmoid()
        
    def forward(self, x):
        """
        Generate mask from input.
        
        Args:
            x: Input tensor [B, T, N]
        
        Returns:
            mask: Continuous mask [B, T, N] with values in [0, 1]
        """
        # x: [B, T, N] -> [B, N, T] for conv1d
        x = x.permute(0, 2, 1)
        
        # Generate mask
        mask = self.encoder(x)
        mask = self.output_activation(mask)
        
        # Back to [B, T, N]
        mask = mask.permute(0, 2, 1)
        
        return mask
    
    def threshold(self, mask, threshold=0.5):
        """
        Binarize mask using threshold.
        
        Args:
            mask: Continuous mask [B, T, N]
            threshold: Threshold value
        
        Returns:
            Binary mask [B, T, N]
        """
        return (mask > threshold).float()
    
    def get_sparsity(self, mask):
        """Calculate actual sparsity of mask (fraction of 1s)."""
        return torch.mean(mask)


class MaskGeneratorTransformer(nn.Module):
    """
    Transformer-based mask generator for more complex patterns.
    """
    
    def __init__(self, seq_len, n_features, d_model=64, n_heads=4, n_layers=2, mask_density=0.032):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.mask_density = mask_density
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_features),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T, N]
        Returns:
            mask: [B, T, N]
        """
        # Project to d_model
        x = self.input_proj(x)  # [B, T, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer encoding
        x = self.transformer(x)  # [B, T, d_model]
        
        # Project to mask
        mask = self.output_proj(x)  # [B, T, N]
        
        return mask
    
    def threshold(self, mask, threshold=0.5):
        return (mask > threshold).float()
    
    def get_sparsity(self, mask):
        return torch.mean(mask)


def create_mask_generator(args, use_transformer=False):
    """
    Factory function to create mask generator.
    
    Args:
        args: Configuration with seq_len, enc_in, etc.
        use_transformer: Use transformer-based or CNN-based generator
    """
    seq_len = args.seq_len
    n_features = args.enc_in
    mask_density = getattr(args, 'mask_density', 0.032)
    
    if use_transformer:
        d_model = getattr(args, 'mask_d_model', 64)
        n_heads = getattr(args, 'mask_n_heads', 4)
        n_layers = getattr(args, 'mask_n_layers', 2)
        return MaskGeneratorTransformer(
            seq_len, n_features, d_model, n_heads, n_layers, mask_density
        )
    else:
        hidden_dim = getattr(args, 'mask_hidden_dim', 64)
        return MaskGenerator(seq_len, n_features, hidden_dim, mask_density)
