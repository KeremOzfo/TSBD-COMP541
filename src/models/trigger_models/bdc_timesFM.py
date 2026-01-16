import torch
from torch import nn
import timesfm


class Model(nn.Module):
    """
    Conditional TimesFM-based Trigger Generator
    
    Uses label embeddings to condition the trigger generation on target classes,
    enabling class-specific backdoor triggers based on Google's foundation model.
    """
    
    def __init__(self, configs, freeze_backbone: bool = True, unfreeze_last_n_layers: int = 2):
        """
        Google's foundation model for zero-shot time series forecasting.
        Customized for conditional backdoor trigger generation.

        Args:
            configs: model configuration object
            freeze_backbone: if True, freeze most of the pretrained model
            unfreeze_last_n_layers: number of final transformer layers to keep trainable
        """
        super().__init__()

        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=configs.seq_len,
                max_horizon=configs.pred_len,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.clip_ratio = getattr(configs, 'clip_ratio', 0.1)
        self.num_class = configs.num_class  # Number of classes for label embedding

        # Freeze/unfreeze layers for efficient fine-tuning
        if freeze_backbone:
            self._freeze_backbone(unfreeze_last_n_layers)

        # Label embedding for conditional generation
        self.label_embed_dim = 64
        self.label_embedding = nn.Embedding(self.num_class, self.label_embed_dim)

        # Trainable output projection head for conditional trigger generation
        hidden_dim = self._get_hidden_dim()
        self.trigger_head = nn.Sequential(
            nn.Linear(hidden_dim + self.label_embed_dim, hidden_dim),  # Concat label embedding
            nn.GELU(),
            nn.Linear(hidden_dim, configs.enc_in),  # output channels
        )

    def _get_hidden_dim(self) -> int:
        """Infer hidden dimension from the pretrained model."""
        return self.enc_in

    def _freeze_backbone(self, unfreeze_last_n: int = 2):
        """
        Freeze the pretrained TimesFM backbone except for:
        - The last `unfreeze_last_n` transformer layers
        - Layer norms (often beneficial to keep trainable)
        - Output/head layers
        """
        # First, freeze everything
        for param in self.model.model.parameters():
            param.requires_grad = False

        # Identify transformer layers
        transformer_layers = []
        for name, module in self.model.model.named_modules():
            if any(pattern in name.lower() for pattern in ['layer', 'block', 'transformer']):
                if hasattr(module, 'parameters') and list(module.parameters()):
                    is_leaf_layer = not any(
                        name != n and n.startswith(name + '.') and 'layer' in n.lower()
                        for n, _ in self.model.model.named_modules()
                    )
                    if is_leaf_layer or 'layer' in name.split('.')[-1].lower():
                        transformer_layers.append((name, module))

        # Deduplicate
        seen_prefixes = set()
        unique_layers = []
        for name, module in transformer_layers:
            layer_id = name
            if layer_id not in seen_prefixes:
                unique_layers.append((name, module))
                seen_prefixes.add(layer_id)

        # Sort by layer index
        def extract_idx(name):
            import re
            nums = re.findall(r'\d+', name)
            return int(nums[-1]) if nums else 0
        unique_layers.sort(key=lambda x: extract_idx(x[0]))

        # Unfreeze last N layers
        if unfreeze_last_n > 0 and unique_layers:
            for name, module in unique_layers[-unfreeze_last_n:]:
                for param in module.parameters():
                    param.requires_grad = True
                print(f"[bdc_timesFM] Unfroze layer: {name}")

        # Always unfreeze LayerNorms
        for name, module in self.model.model.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
                for param in module.parameters():
                    param.requires_grad = True

        # Unfreeze output/head layers
        if unfreeze_last_n > 0:
            for name, module in self.model.model.named_modules():
                if any(kw in name.lower() for kw in ['head', 'output', 'proj', 'fc', 'classifier', 'quantile']):
                    for param in module.parameters():
                        param.requires_grad = True
                    print(f"[bdc_timesFM] Unfroze head/output: {name}")

        # Report trainable params
        total = sum(p.numel() for p in self.model.model.parameters())
        trainable = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        print(f"[bdc_timesFM] Backbone: {trainable:,} / {total:,} params trainable ({100*trainable/total:.2f}%)")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Run TimesFM forecast and return raw output."""
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        B, L, C = x_enc.shape
        device = x_enc.device
        x_enc_flat = torch.reshape(x_enc, (B*C, L))

        output, _ = self.model.forecast(
            horizon=self.pred_len,
            inputs=x_enc_flat.cpu().numpy()
        )
        output = torch.Tensor(output).to(device)

        dec_out = torch.reshape(output, (B, output.shape[-1], C)).to(device)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, bd_labels=None):
        """
        Forward pass for conditional trigger generation.

        Args:
            x_enc: B x T x N input time series
            bd_labels: B tensor of target class labels for conditioning
        Returns: 
            (trigger, clipped_trigger)
        """
        if bd_labels is None:
            # Default to class 0 if no labels provided
            bd_labels = torch.zeros(x_enc.shape[0], dtype=torch.long, device=x_enc.device)
        
        # Get base forecast from frozen/partially-frozen TimesFM
        base_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # Get label embeddings: B -> B x label_embed_dim
        # Squeeze bd_labels in case it has shape [B, 1] instead of [B]
        if bd_labels.dim() > 1:
            bd_labels = bd_labels.squeeze(-1)
        label_emb = self.label_embedding(bd_labels)  # B x label_embed_dim
        
        # Expand label embedding to match time dimension: B x T x label_embed_dim
        B, T, C = base_out.shape
        label_cond = label_emb.unsqueeze(1).expand(-1, T, -1)  # B x T x label_embed_dim
        
        # Concatenate label conditioning with base output
        conditioned_input = torch.cat([base_out, label_cond], dim=-1)  # B x T x (C + label_embed_dim)
        
        # Pass through conditional trigger head
        trigger = self.trigger_head(conditioned_input)  # B x T x C

        # Pad or truncate to match input length if needed
        if T != x_enc.shape[1]:
            target_len = x_enc.shape[1]
            if T < target_len:
                pad = torch.zeros(B, target_len - T, C, device=trigger.device)
                trigger = torch.cat([pad, trigger], dim=1)
            else:
                trigger = trigger[:, -target_len:, :]

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


if __name__ == "__main__":
    """Test the bdc_timesFM model with dummy data."""
    
    class DummyConfigs:
        """Minimal config for testing."""
        def __init__(self):
            self.seq_len = 128
            self.pred_len = 32
            self.enc_in = 3
            self.task_name = "classification"
            self.clip_ratio = 0.1
            self.num_class = 8  # Example: 8 classes

    print("=" * 60)
    print("Testing bdc_timesFM Conditional Backdoor Trigger Generator")
    print("=" * 60)

    configs = DummyConfigs()
    print(f"\nConfig: seq_len={configs.seq_len}, pred_len={configs.pred_len}, enc_in={configs.enc_in}, num_class={configs.num_class}")

    print("\n[Test 1] Creating model...")
    try:
        model = Model(configs, freeze_backbone=True, unfreeze_last_n_layers=0)
        print("[Test 1] ✓ Model created successfully")
    except Exception as e:
        print(f"[Test 1] ✗ Failed to create model: {e}")
        exit(1)

    print("\n[Test 2] Running forward pass with labels...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    batch_size = 4
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in).to(device)
    bd_labels = torch.randint(0, configs.num_class, (batch_size,)).to(device)
    print(f"Input shape: {x_enc.shape}, Labels: {bd_labels.tolist()}")

    try:
        with torch.no_grad():
            trigger, clipped = model(x_enc, None, None, None, bd_labels=bd_labels)
        print(f"Trigger shape: {trigger.shape}")
        print(f"Clipped trigger shape: {clipped.shape}")
        print("[Test 2] ✓ Forward pass successful")
    except Exception as e:
        print(f"[Test 2] ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
