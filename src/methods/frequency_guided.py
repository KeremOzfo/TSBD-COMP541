import torch
import torch.nn as nn
import numpy as np

from .utils import cal_accuracy



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


def epoch_frequency_guided(
    bd_model,
    surr_model,
    loader,
    args,
    opt=None,
    train: bool = True,
):
    """Frequency-domain guided trigger training"""

    # Hyperparameters
    alpha = getattr(args, "freq_alpha", 1.0)  # weight for L_Freq
    beta = getattr(args, "freq_beta", 1e-3)   # weight for regularization
    heatmap_batches = getattr(args, "freq_heatmap_batches", 1)

    # Pre-compute frequency sensitivity heatmap once
    freq_target = estimate_frequency_heatmap(surr_model, loader, args, max_batches=heatmap_batches)
    freq_target = freq_target.detach()  # (freq_len, channels)

    total_loss = []
    loss_dict = {"CE_bd": [], "L_freq": [], "L_reg": []}
    bd_preds = []
    trues = []
    bd_labels_all = []

    if train:
        bd_model.train()
        surr_model.train()
    else:
        bd_model.eval()
        surr_model.eval()

    freq_len = freq_target.shape[0]

    for batch_x, label, padding_mask in loader:
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device).long()

        # target labels based on bd_type
        bd_type = getattr(args, 'bd_type', 'all2one')
        num_classes = getattr(args, 'num_class', getattr(args, 'numb_class', None))
        if bd_type == 'all2all' and num_classes is not None:
            bd_labels = torch.randint(0, num_classes, label.shape, device=args.device)
        else:  # all2one
            bd_labels = torch.ones_like(label).to(args.device) * args.target_label

        # mask if attacking only non-target
        if getattr(args, "attack_only_nontarget", False):
            attack_mask = (label != args.target_label).float().to(args.device)
        else:
            attack_mask = torch.ones_like(label).float().to(args.device)
        attack_mask = attack_mask.unsqueeze(-1).expand_as(batch_x)

        # Generate trigger
        trigger, trigger_clip = bd_model(batch_x, padding_mask, None, None, bd_labels)
        triggered_inputs = batch_x + trigger_clip * attack_mask

        # Classification loss on triggered inputs
        pred_bd = surr_model(triggered_inputs, padding_mask, None, None)
        loss_ce = args.criterion(pred_bd, bd_labels.long().squeeze(-1))

        # Frequency alignment loss (Eq. 3): compare spectrum magnitude with sensitivity heatmap
        trigger_freq = torch.fft.rfft(trigger_clip, dim=1)
        trigger_mag = torch.abs(trigger_freq)
        target_mag = freq_target[: trigger_mag.shape[1]].to(args.device)
        loss_freq = torch.mean((trigger_mag - target_mag) ** 2)

        # Regularization in both frequency and temporal domains (Eq. 4)
        loss_reg = torch.mean(trigger_mag ** 2)
        loss = loss_ce + alpha * loss_freq + beta * loss_reg

        if opt is not None and train:
            opt.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if hasattr(args, 'trigger_grad_clip') and args.trigger_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(bd_model.parameters(), args.trigger_grad_clip)
            
            opt.step()

        total_loss.append(loss.item())
        loss_dict["CE_bd"].append(loss_ce.item())
        loss_dict["L_freq"].append(loss_freq.item())
        loss_dict["L_reg"].append(loss_reg.item())

        bd_preds.append(pred_bd)
        trues.append(label)
        bd_labels_all.append(bd_labels)

    if not total_loss:
        return 0.0, loss_dict, 0.0, 0.0

    total_loss = float(np.average(total_loss))
    bd_preds = torch.cat(bd_preds, 0)
    trues = torch.cat(trues, 0)
    bd_labels_all = torch.cat(bd_labels_all, 0)

    bd_predictions = torch.argmax(torch.nn.functional.softmax(bd_preds, dim=1), dim=1).cpu().numpy()
    trues_np = trues.flatten().cpu().numpy()
    bd_labels_np = bd_labels_all.flatten().cpu().numpy()

    # Clean accuracy is not directly optimized here; report ASR as main metric
    bd_accuracy = cal_accuracy(bd_predictions, bd_labels_np)
    clean_accuracy = cal_accuracy(bd_predictions, trues_np)

    return total_loss, loss_dict, clean_accuracy, bd_accuracy