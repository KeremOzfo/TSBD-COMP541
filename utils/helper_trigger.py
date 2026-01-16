def _frequency_basis(seq_len: int, channels: int, device: torch.device, cache: dict, t: int) -> torch.Tensor:
    """Return a real-valued temporal basis whose spectrum has energy only at freq bin t."""
    if t in cache:
        return cache[t]

    freq_len = seq_len // 2 + 1
    freq_mask = torch.zeros(freq_len, dtype=torch.cfloat, device=device)
    # place symmetric energy at bin t (0-indexed); irfft handles the conjugate symmetry
    if t < freq_len:
        freq_mask[t] = torch.tensor(1.0, dtype=torch.cfloat, device=device)
    basis = torch.fft.irfft(freq_mask, n=seq_len).real  # (seq_len,)
    basis = basis.view(1, seq_len, 1).expand(1, seq_len, channels)  # broadcast across channels
    cache[t] = basis
    return basis


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

