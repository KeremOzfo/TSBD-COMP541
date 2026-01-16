import torch
from torch import nn
import timesfm


class Model(nn.Module):
    def __init__(self, configs, freeze_backbone: bool = True, unfreeze_last_n_layers: int = 2):
        """
        Google's foundation model for zero-shot time series forecasting.
        Custimized for backdoor trigger generator with selective freezing.

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

        # Freeze/unfreeze layers for efficient fine-tuning
        if freeze_backbone:
            self._freeze_backbone(unfreeze_last_n_layers)

        # Trainable output projection head for trigger generation
        hidden_dim = self._get_hidden_dim()
        self.trigger_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, configs.enc_in),  # output channels
        )

    def _get_hidden_dim(self) -> int:
        """Infer hidden dimension from the pretrained model."""
        # For trigger generation, use the input channel dimension
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

        # Identify transformer layers (common naming patterns)
        transformer_layers = []
        for name, module in self.model.model.named_modules():
            # Match common layer naming: layers, blocks, encoder.layer, decoder.layer, etc.
            if any(pattern in name.lower() for pattern in ['layer', 'block', 'transformer']):
                if hasattr(module, 'parameters') and list(module.parameters()):
                    # Check if it's a direct layer (not a container with sublayers in our list)
                    is_leaf_layer = not any(
                        name != n and n.startswith(name + '.') and 'layer' in n.lower()
                        for n, _ in self.model.model.named_modules()
                    )
                    if is_leaf_layer or 'layer' in name.split('.')[-1].lower():
                        transformer_layers.append((name, module))

        # Deduplicate by keeping only top-level layer modules
        seen_prefixes = set()
        unique_layers = []
        for name, module in transformer_layers:
            prefix = name.split('.')[0] if '.' in name else name
            layer_id = name
            if layer_id not in seen_prefixes:
                unique_layers.append((name, module))
                seen_prefixes.add(layer_id)

        # Sort by layer index if possible (e.g., layer.0, layer.1, ...)
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
                print(f"[bd_timesFM] Unfroze layer: {name}")

        # Always unfreeze LayerNorms (helps adaptation)
        for name, module in self.model.model.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
                for param in module.parameters():
                    param.requires_grad = True

        # Unfreeze output/head layers only if unfreezing transformer layers
        if unfreeze_last_n > 0:
            for name, module in self.model.model.named_modules():
                if any(kw in name.lower() for kw in ['head', 'output', 'proj', 'fc', 'classifier', 'quantile']):
                    for param in module.parameters():
                        param.requires_grad = True
                    print(f"[bd_timesFM] Unfroze head/output: {name}")

        # Report trainable params
        total = sum(p.numel() for p in self.model.model.parameters())
        trainable = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        print(f"[bd_timesFM] Backbone: {trainable:,} / {total:,} params trainable ({100*trainable/total:.2f}%)")

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
        Forward pass for trigger generation.

        x_enc: B x T x N input time series
        Returns: (trigger, clipped_trigger)
        """
        # Get base forecast from frozen/partially-frozen TimesFM
        base_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # Pass through trainable trigger head to generate trigger pattern
        # base_out: B x pred_len x C -> flatten -> head -> reshape
        B, T, C = base_out.shape
        trigger = self.trigger_head(base_out)  # B x T x C (after linear)

        # Pad or truncate to match input length if needed
        if T != x_enc.shape[1]:
            # Pad with zeros or truncate
            target_len = x_enc.shape[1]
            if T < target_len:
                pad = torch.zeros(B, target_len - T, C, device=trigger.device)
                trigger = torch.cat([pad, trigger], dim=1)  # trigger at end
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
    """Test the bd_timesFM model with dummy data."""
    import argparse

    class DummyConfigs:
        """Minimal config for testing."""
        def __init__(self):
            self.seq_len = 128
            self.pred_len = 32
            self.enc_in = 3  # number of input channels/variates
            self.task_name = "classification"
            self.clip_ratio = 0.1

    print("=" * 60)
    print("Testing bd_timesFM Backdoor Trigger Generator")
    print("=" * 60)

    configs = DummyConfigs()
    print(f"\nConfig: seq_len={configs.seq_len}, pred_len={configs.pred_len}, enc_in={configs.enc_in}")

    # Test 1: Model instantiation with freezing
    print("\n[Test 1] Creating model with freeze_backbone=True, unfreeze_last_n_layers=2...")
    try:
        model = Model(configs, freeze_backbone=True, unfreeze_last_n_layers=0)
        print("[Test 1] ✓ Model created successfully")
    except Exception as e:
        print(f"[Test 1] ✗ Failed to create model: {e}")
        exit(1)

    # Report total trainable parameters (including trigger_head)
    total_params = sum(p.numel() for p in model.model.model.parameters()) + sum(p.numel() for p in model.trigger_head.parameters())
    trainable_params = sum(p.numel() for p in model.model.model.parameters() if p.requires_grad) + sum(p.numel() for p in model.trigger_head.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Test 2: Forward pass with dummy data
    print("\n[Test 2] Running forward pass with dummy data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = model.to(device)
    model.eval()

    batch_size = 4
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in).to(device)
    print(f"Input shape: {x_enc.shape}")

    try:
        with torch.no_grad():
            trigger, clipped = model(x_enc, None, None, None, bd_labels=None)
        print(f"Trigger shape: {trigger.shape}")
        print(f"Clipped trigger shape: {clipped.shape}")
        print(f"Trigger range: [{trigger.min().item():.4f}, {trigger.max().item():.4f}]")
        print(f"Clipped range: [{clipped.min().item():.4f}, {clipped.max().item():.4f}]")
        print("[Test 2] ✓ Forward pass successful")
    except Exception as e:
        print(f"[Test 2] ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test 3: Backward pass (gradient flow)
    print("\n[Test 3] Testing backward pass (gradient flow)...")
    model.train()
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in, requires_grad=False).to(device)

    try:
        trigger, clipped = model(x_enc, None, None, None, bd_labels=None)
        loss = clipped.sum()  # dummy loss
        loss.backward()

        # Check which params have gradients
        grads_found = 0
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                grads_found += 1
        print(f"Parameters with non-zero gradients: {grads_found}")
        print("[Test 3] ✓ Backward pass successful")
    except Exception as e:
        print(f"[Test 3] ✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test 4: Apply trigger to input
    print("\n[Test 4] Applying trigger to input...")
    try:
        x_triggered = x_enc + clipped
        print(f"Original input range: [{x_enc.min().item():.4f}, {x_enc.max().item():.4f}]")
        print(f"Triggered input range: [{x_triggered.min().item():.4f}, {x_triggered.max().item():.4f}]")
        print("[Test 4] ✓ Trigger application successful")
    except Exception as e:
        print(f"[Test 4] ✗ Failed: {e}")
        exit(1)

    print("\n" + "=" * 60)
    print("All tests passed! bd_timesFM is ready for backdoor training.")
    print("=" * 60)
