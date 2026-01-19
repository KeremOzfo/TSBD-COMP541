import torch
import torch.nn as nn

def diversity_loss(x1, x2, g_x1, g_x2, epsilon=1e-6):
    """
    Compute the diversity enforcement regularization loss.
    
    L_div = ||x - x'|| / ||g(x) - g(x')||
    
    This loss encourages the trigger generator to produce diverse triggers
    for different inputs, preventing saturation to a uniform trigger.
    
    Args:
        x1: First batch of clean inputs (B x T x C or B x C x T)
        x2: Second batch of clean inputs (different from x1)
        g_x1: Triggers generated for x1
        g_x2: Triggers generated for x2
        epsilon: Constant for numerical stability
    
    Returns:
        Scalar diversity loss value
    """
    # Compute input distances: ||x - x'||
    input_diff = x1 - x2
    input_distances = torch.sqrt(torch.mean(input_diff ** 2, dim=(1, 2)) + epsilon)
    
    # Compute trigger distances: ||g(x) - g(x')||
    trigger_diff = g_x1 - g_x2
    trigger_distances = torch.sqrt(torch.mean(trigger_diff ** 2, dim=(1, 2)) + epsilon)
    
    # L_div = ||x - x'|| / ||g(x) - g(x')||
    # We want to minimize this, which means maximizing trigger diversity
    loss_div = input_distances / (trigger_distances + epsilon)
    
    return torch.mean(loss_div)


def estimate_frequency_heatmap(model, loader, args, max_batches: int | None = None) -> torch.Tensor:
    """Estimate model sensitivity heatmap S_{t,m} via DFT-based perturbations"""
    model.eval()
    device = args.device
    seq_len = args.seq_len
    perturb_scale = getattr(args, "freq_lambda", 0.05)
    max_bins = getattr(args, "freq_max_bins", seq_len)
    freq_len = seq_len // 2 + 1

    heatmap = None
    basis_cache: dict[int, torch.Tensor] = {}
    batches_used = 0

    with torch.no_grad():
        for b_idx, (batch_x, label, padding_mask) in enumerate(loader):
            if max_batches is not None and batches_used >= max_batches:
                break
            batch_x = batch_x.float().to(device)
            padding_mask = padding_mask.float().to(device)
            label = label.to(device)

            if heatmap is None:
                channels = batch_x.shape[-1]
                heatmap = torch.zeros(freq_len, channels, device=device)

            # base loss
            base_pred = model(batch_x, padding_mask, None, None)
            base_loss = args.criterion(base_pred, label.long().squeeze(-1))

            for t in range(min(freq_len, max_bins)):
                basis = _frequency_basis(seq_len, batch_x.shape[-1], device, basis_cache, t)
                perturbed = batch_x + perturb_scale * basis
                pert_pred = model(perturbed, padding_mask, None, None)
                pert_loss = args.criterion(pert_pred, label.long().squeeze(-1))
                delta = (pert_loss - base_loss).detach()
                heatmap[t] += delta

            batches_used += 1

    if heatmap is None:
        # empty loader fallback
        heatmap = torch.zeros(seq_len // 2 + 1, getattr(args, "enc_in", 1), device=device)
    else:
        heatmap = heatmap / max(1, batches_used)

    return heatmap.detach()

def fftreg(x_clean, x_back):
    """
    Frequency-Domain Regularization Loss (FFT-based)
    
    Measures the spectral divergence between clean and backdoored data to ensure
    triggers are frequency-stealthy (minimizing frequency-domain artifacts).
    
    Mathematical formulation:
    
    Given:
    - x_clean ∈ ℝ^(B×T×C): Clean time series (Batch size, Time steps, Channels)
    - x_back ∈ ℝ^(B×T×C): Backdoored time series
    
    Process:
    1. Permute to B×C×T format: x_c, x_b = Permute(x_clean, x_back)
    
    2. Compute FFT (Real FFT):
       X_c = |FFT(x_c)| ∈ ℝ^(B×C×F)  where F = T/2 + 1 (frequency bins)
       X_b = |FFT(x_b)| ∈ ℝ^(B×C×F)
    
    3. Remove DC component (frequency 0):
       X̂_c = X_c[:, :, 1:-1]  # Exclude f=0 to focus on oscillatory behavior
       X̂_b = X_b[:, :, 1:-1]
    
    4. Compute Cosine Similarity (normalized dot product):
       L_fft = (1/B) Σ_b (1/C) Σ_c cos_sim(X̂_c[b,c,:], X̂_b[b,c,:])
       
       where cos_sim(u, v) = (u·v) / (||u||₂ · ||v||₂)
    
    Purpose:
    - Maximizes L_fft ≈ 1: backdoored and clean data have similar frequency spectra
    - Minimizes frequency-domain differences to reduce detectability
    - Prevents triggers from introducing sharp spectral peaks
    
    Args:
        x_clean: Clean input time series (B × T × C)
        x_back: Backdoored input time series (B × T × C)
    
    Returns:
        Scalar loss ∈ [-1, 1]: Average cosine similarity across all batch & channel pairs
        (Higher values indicate frequency stealthiness)
    """
    # Permute from B x T x C to B x C x T for FFT computation
    x_c = x_clean.float().permute(0, 2, 1)
    x_b = x_back.float().permute(0, 2, 1)
    
    # Initialize cosine similarity metric
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    
    # Compute FFT magnitude spectra
    xf_c = abs(torch.fft.rfft(x_c, dim=2))  # Clean data frequency domain
    xf_b = abs(torch.fft.rfft(x_b, dim=2))  # Backdoored data frequency domain
    
    # Remove DC component (frequency bin 0) to focus on oscillations
    xf_c2 = xf_c[:, :, 1:-1]
    xf_b2 = xf_b[:, :, 1:-1]
    
    # Return average cosine similarity across batch and channels
    return cos(xf_c2, xf_b2).mean()
 