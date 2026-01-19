"""
Combined Frequency-Guided + Marksman Epoch Training

This module combines:
1. Frequency-guided trigger training (epoch_frequency_guided) - uses DFT-based sensitivity heatmap
2. Marksman input-aware training (epoch_marksman_input_aware) - uses diversity and cross-trigger losses

All loss functions are parameterized except CrossEntropyLoss weight=1.

Loss Components:
    Classifier update (Phase 1):
        L_class = CE_clean + alpha * CE_attack + lambda_cross * CE_cross
    
    Trigger update (Phase 2):
        L_trigger = CE_bd + lambda_freq * L_freq + lambda_div * L_div + lambda_reg * L_reg - beta * ||g||^2

Parameters:
    - alpha: Weight for backdoor CE in classifier update (Marksman)
    - beta: Weight for trigger magnitude penalty (Marksman)
    - update_T: Trigger update frequency - every T steps (Marksman)
    - lambda_freq: Weight for frequency alignment loss
    - lambda_div: Weight for diversity loss  
    - lambda_reg: Weight for regularization loss
    - lambda_cross: Weight for cross-trigger loss
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class CombinedEpochParams:
    """Parameters for combined frequency-guided + marksman training.
    
    All loss weights are configurable except CE which is always weight=1.0.
    """
    # Target label for backdoor attack
    target_label: int = 0
    
    # Marksman mode splitting ratios
    p_attack: float = 0.5       # rho_b: fraction of batch for backdoor samples
    p_cross: float = 0.1        # rho_c: fraction of batch for cross-trigger samples
    
    # Marksman parameters (T, alpha, beta)
    marksman_update_T: int = 1  # Trigger update frequency (every T steps)
    marksman_alpha: float = 1.0 # Weight for BD CE loss in classifier update
    marksman_beta: float = 0.0  # Weight for trigger magnitude penalty
    
    # Loss weights (all parameterized except CE)
    lambda_freq: float = 1.0    # Weight for frequency alignment loss
    lambda_div: float = 1.0     # Weight for diversity loss
    lambda_reg: float = 1e-3    # Weight for regularization loss (temporal + frequency)
    lambda_cross: float = 1.0   # Weight for cross-trigger nonreusability loss
    
    # Frequency estimation settings
    freq_heatmap_batches: int = 1       # Batches to use for heatmap estimation
    freq_lambda: float = 0.05           # Perturbation scale for frequency estimation
    freq_max_bins: int = 256            # Maximum frequency bins
    
    # Training behavior
    attack_only_nontarget: bool = False  # Only attack non-target samples
    bd_type: str = 'all2one'    # 'all2one' or 'all2all'


def cal_accuracy(preds, trues):
    """Compute accuracy given predicted labels and true labels."""
    preds_np = np.asarray(preds).reshape(-1)
    trues_np = np.asarray(trues).reshape(-1)
    if preds_np.shape[0] == 0:
        return 0.0
    return float(np.mean(preds_np == trues_np))


def pull_model(dst, src):
    """Copy model state from src to dst."""
    dst.load_state_dict(src.state_dict())


def diversity_loss(x1, x2, g_x1, g_x2, epsilon=1e-6):
    """Compute diversity enforcement loss L_div = ||x - x'|| / ||g(x) - g(x')||.
    
    Encourages trigger generator to produce diverse triggers for different inputs.
    """
    input_diff = x1 - x2
    input_distances = torch.sqrt(torch.mean(input_diff ** 2, dim=(1, 2)) + epsilon)
    
    trigger_diff = g_x1 - g_x2
    trigger_distances = torch.sqrt(torch.mean(trigger_diff ** 2, dim=(1, 2)) + epsilon)
    
    loss_div = input_distances / (trigger_distances + epsilon)
    return torch.mean(loss_div)


def apply_trigger(x, trigger, mask=None):
    """Apply trigger to input: B(x, t) = x + trigger."""
    if mask is not None:
        return x + trigger * mask
    return x + trigger


def _frequency_basis(seq_len: int, channels: int, device: torch.device, 
                     cache: dict, t: int) -> torch.Tensor:
    """Return a real-valued temporal basis whose spectrum has energy only at freq bin t."""
    if t in cache:
        return cache[t]

    freq_len = seq_len // 2 + 1
    freq_mask = torch.zeros(freq_len, dtype=torch.cfloat, device=device)
    if t < freq_len:
        freq_mask[t] = torch.tensor(1.0, dtype=torch.cfloat, device=device)
    basis = torch.fft.irfft(freq_mask, n=seq_len).real
    basis = basis.view(1, seq_len, 1).expand(1, seq_len, channels)
    cache[t] = basis
    return basis


def estimate_frequency_heatmap(model, loader, args, max_batches: int = 1) -> torch.Tensor:
    """Estimate model sensitivity heatmap S_{t,m} via DFT-based perturbations."""
    model.eval()
    device = args.device
    seq_len = args.seq_len
    perturb_scale = getattr(args, "freq_lambda", 0.05)
    max_bins = getattr(args, "freq_max_bins", seq_len)
    freq_len = seq_len // 2 + 1

    heatmap = None
    basis_cache: dict = {}
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
        heatmap = torch.zeros(seq_len // 2 + 1, getattr(args, "enc_in", 1), device=device)
    else:
        heatmap = heatmap / max(1, batches_used)

    return heatmap.detach()


def compute_frequency_loss(trigger_clip, freq_target, device):
    """Compute frequency alignment loss between trigger and sensitivity heatmap.
    
    Args:
        trigger_clip: Clipped trigger tensor (B, T, C)
        freq_target: Frequency sensitivity heatmap (freq_len, C)
        device: Torch device
        
    Returns:
        Scalar frequency alignment loss
    """
    trigger_freq = torch.fft.rfft(trigger_clip, dim=1)
    trigger_mag = torch.abs(trigger_freq)
    target_mag = freq_target[:trigger_mag.shape[1]].to(device)
    loss_freq = torch.mean((trigger_mag - target_mag) ** 2)
    return loss_freq


def compute_regularization_loss(trigger, trigger_clip):
    """Compute combined frequency and temporal regularization.
    
    Args:
        trigger: Raw trigger tensor
        trigger_clip: Clipped trigger tensor
        
    Returns:
        Scalar regularization loss
    """
    trigger_freq = torch.fft.rfft(trigger_clip, dim=1)
    trigger_mag = torch.abs(trigger_freq)
    loss_reg = torch.mean(trigger_mag ** 2) + torch.mean(trigger_clip ** 2)
    return loss_reg


def epoch_combined(
    trigger_model,
    trigger_model_prev,
    surrogate_model,
    loader1,
    args,
    loader2=None,
    opt_trig=None,
    opt_class=None,
    train: bool = True,
    freq_target: Optional[torch.Tensor] = None,
):
    """Combined frequency-guided + marksman input-aware training epoch.
    
    This function combines:
    1. Frequency-guided loss: align triggers to model sensitivity heatmap
    2. Marksman framework: classifier + trigger alternating updates
    3. Input-aware diversity: prevent trigger saturation
    4. Cross-trigger nonreusability: triggers cannot transfer between inputs
    
    Loss Function:
        L_class = CE_clean + CE_attack + lambda_cross * CE_cross
        L_trigger = CE_bd + lambda_freq * L_freq + lambda_div * L_div + lambda_reg * L_reg - beta * ||g||^2
    
    Args:
        trigger_model: Trigger generator to train
        trigger_model_prev: Frozen trigger generator for stable classifier training
        surrogate_model: Classifier network
        loader1: Primary data loader
        args: Training arguments (should include CombinedEpochParams fields or defaults)
        loader2: Optional secondary loader for diversity (uses shuffled batch if None)
        opt_trig: Optimizer for trigger generator
        opt_class: Optimizer for classifier
        train: Training mode flag
        freq_target: Pre-computed frequency heatmap (computed if None)
        
    Returns:
        total_loss: Average total loss
        clean_accuracy: Accuracy on clean samples
        bd_accuracy: Attack success rate
    """
    # Extract parameters with defaults
    target_label = getattr(args, 'target_label', 0)
    rho_b = getattr(args, 'p_attack', 0.5)
    rho_c = getattr(args, 'p_cross', 0.1)
    lambda_freq = getattr(args, 'lambda_freq', 1.0)
    lambda_div = getattr(args, 'lambda_div', 1.0)
    lambda_reg = getattr(args, 'lambda_reg', 1e-3)
    lambda_cross = getattr(args, 'lambda_cross', 1.0)
    # Marksman parameters
    alpha = getattr(args, 'marksman_alpha', 1.0)  # Weight for BD loss in classifier update
    beta = getattr(args, 'marksman_beta', 0.0)    # Weight for trigger magnitude penalty
    update_T = getattr(args, 'marksman_update_T', 1)  # Trigger update frequency
    bd_type = getattr(args, 'bd_type', 'all2one')
    num_classes = getattr(args, 'num_class', getattr(args, 'numb_class', None))
    
    assert rho_b + rho_c < 1.0, "p_attack + p_cross must be less than 1"
    
    # Estimate frequency heatmap if not provided
    if freq_target is None:
        heatmap_batches = getattr(args, 'freq_heatmap_batches', 1)
        freq_target = estimate_frequency_heatmap(surrogate_model, loader1, args, max_batches=heatmap_batches)
    
    # Setup loaders
    if loader2 is None:
        loader2 = [[None, None, None]] * len(loader1)
    
    # Set model modes
    trigger_model_prev.eval()
    if train:
        surrogate_model.train()
        trigger_model.train()
    else:
        surrogate_model.eval()
        trigger_model.eval()
    
    # Tracking variables
    total_loss = []
    loss_dict = {
        'CE_clean': [], 'CE_bd': [], 'CE_cross': [],
        'L_freq': [], 'L_div': [], 'L_reg': [], 'L_trig': []
    }
    clean_preds, bd_preds, cross_preds = [], [], []
    clean_labels, bd_labels_all, cross_labels = [], [], []
    
    for step, ((batch_x, label, padding_mask), (batch_x2, label2, padding_mask2)) in enumerate(
        zip(loader1, loader2)
    ):
        batch_size = batch_x.shape[0]
        
        # Move data to device
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device).long()
        
        # Handle second batch
        if batch_x2 is not None:
            batch_x2 = batch_x2.float().to(args.device)
            padding_mask2 = padding_mask2.float().to(args.device)
            label2 = label2.to(args.device)
        else:
            indices = torch.randperm(batch_size)
            batch_x2 = batch_x[indices]
            padding_mask2 = padding_mask[indices]
            label2 = label[indices]
        
        # Generate backdoor target labels
        if bd_type == 'all2all' and num_classes is not None:
            bd_labels = torch.randint(0, num_classes, label.shape, device=args.device)
        else:  # all2one
            bd_labels = torch.ones_like(label).to(args.device) * target_label
        
        # Compute batch split sizes
        num_bd = max(1, int(rho_b * batch_size))
        num_cross = max(1, min(int(rho_c * batch_size), batch_size - num_bd - 1))
        num_clean = batch_size - num_bd - num_cross
        
        # Create attack mask
        if getattr(args, 'attack_only_nontarget', False):
            attack_mask = (label != target_label).float().to(args.device)
        else:
            attack_mask = torch.ones(batch_size, device=args.device).float()
        attack_mask = attack_mask.unsqueeze(-1).unsqueeze(-1)
        if len(batch_x.shape) == 3:
            attack_mask = attack_mask.squeeze(-1).expand(-1, batch_x.shape[1], batch_x.shape[2])
        
        # ================================================================
        # PHASE 1: Train Classifier (with frozen trigger generator)
        # Marksman: L_class = CE_clean + alpha * CE_attack + lambda_cross * CE_cross
        # ================================================================
        if opt_class is not None:
            opt_class.zero_grad()
        trigger_model.zero_grad()
        
        with torch.no_grad():
            trigger_t, trigger_t_clip = trigger_model_prev(batch_x, padding_mask, None, None, bd_labels)
            trigger_t_prime, trigger_t_prime_clip = trigger_model_prev(batch_x2, padding_mask2, None, None, bd_labels)
        
        # Split batches for three modes
        x_attack = batch_x[:num_bd]
        y_attack = bd_labels[:num_bd]
        trigger_attack = trigger_t_clip[:num_bd]
        mask_attack = attack_mask[:num_bd]
        
        x_cross = batch_x[num_bd:num_bd + num_cross]
        y_cross = label[num_bd:num_bd + num_cross]  # Original label for nonreusability!
        trigger_cross = trigger_t_prime_clip[num_bd:num_bd + num_cross]
        mask_cross = attack_mask[num_bd:num_bd + num_cross]
        
        x_clean = batch_x[num_bd + num_cross:]
        y_clean = label[num_bd + num_cross:]
        
        # Forward pass for each mode
        bd_inputs = apply_trigger(x_attack, trigger_attack, mask_attack)
        pred_attack = surrogate_model(bd_inputs, padding_mask[:num_bd], None, None)
        
        cross_inputs = apply_trigger(x_cross, trigger_cross, mask_cross)
        pred_cross = surrogate_model(cross_inputs, padding_mask[num_bd:num_bd + num_cross], None, None)
        
        pred_clean = surrogate_model(x_clean, padding_mask[num_bd + num_cross:], None, None)
        
        # Compute CE losses
        loss_ce_attack = args.criterion(pred_attack, y_attack.long().squeeze(-1))
        loss_ce_cross = args.criterion(pred_cross, y_cross.long().squeeze(-1))
        loss_ce_clean = args.criterion(pred_clean, y_clean.long().squeeze(-1))
        
        # Total classifier loss with Marksman alpha and parameterized cross weight
        # L_class = CE_clean + alpha * CE_attack + lambda_cross * CE_cross
        loss_class = loss_ce_clean + alpha * loss_ce_attack + lambda_cross * loss_ce_cross
        
        if train and opt_class is not None:
            loss_class.backward()
            
            # Gradient clipping
            if hasattr(args, 'surrogate_grad_clip') and args.surrogate_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), args.surrogate_grad_clip)
            
            opt_class.step()
        
        # ================================================================
        # PHASE 2: Train Trigger Generator (with frozen classifier)
        # ================================================================
        loss_trig_val = 0.0
        loss_freq_val = 0.0
        loss_div_val = 0.0
        loss_reg_val = 0.0
        
        if train and opt_trig is not None and (step % update_T == 0):
            # Freeze classifier
            requires_backup = [p.requires_grad for p in surrogate_model.parameters()]
            for p in surrogate_model.parameters():
                p.requires_grad = False
            
            opt_trig.zero_grad()
            
            # Generate triggers with active model
            trigger_t, trigger_t_clip = trigger_model(batch_x, padding_mask, None, None, bd_labels)
            trigger_t_prime, trigger_t_prime_clip = trigger_model(batch_x2, padding_mask2, None, None, bd_labels)
            
            # Backdoor CE loss for trigger
            bd_inputs_trig = apply_trigger(batch_x[:num_bd], trigger_t_clip[:num_bd], mask_attack)
            pred_bd_trig = surrogate_model(bd_inputs_trig, padding_mask[:num_bd], None, None)
            loss_ce_trig = args.criterion(pred_bd_trig, bd_labels[:num_bd].long().squeeze(-1))
            
            # Frequency alignment loss (parameterized by lambda_freq)
            loss_freq = compute_frequency_loss(trigger_t_clip, freq_target, args.device)
            
            # Diversity loss (parameterized by lambda_div)
            loss_div = diversity_loss(batch_x, batch_x2, trigger_t, trigger_t_prime)
            
            # Regularization loss (parameterized by lambda_reg)
            loss_reg = compute_regularization_loss(trigger_t, trigger_t_clip)
            
            # Trigger magnitude penalty (parameterized by beta)
            loss_magnitude = torch.mean(trigger_t ** 2)
            
            # Total trigger loss
            loss_trig = (loss_ce_trig 
                        + lambda_freq * loss_freq 
                        + lambda_div * loss_div 
                        + lambda_reg * loss_reg 
                        - beta * loss_magnitude)
            
            loss_trig.backward()
            
            # Gradient clipping
            if hasattr(args, 'trigger_grad_clip') and args.trigger_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trigger_model.parameters(), args.trigger_grad_clip)
            
            opt_trig.step()
            
            # Restore classifier gradients
            for p, req in zip(surrogate_model.parameters(), requires_backup):
                p.requires_grad = req
            
            loss_trig_val = loss_trig.item()
            loss_freq_val = loss_freq.item()
            loss_div_val = loss_div.item()
            loss_reg_val = loss_reg.item()
        
        # ================================================================
        # Logging
        # ================================================================
        total_loss.append(loss_class.item() + loss_trig_val)
        loss_dict['CE_clean'].append(loss_ce_clean.item())
        loss_dict['CE_bd'].append(loss_ce_attack.item())
        loss_dict['CE_cross'].append(loss_ce_cross.item())
        loss_dict['L_freq'].append(loss_freq_val)
        loss_dict['L_div'].append(loss_div_val)
        loss_dict['L_reg'].append(loss_reg_val)
        loss_dict['L_trig'].append(loss_trig_val)
        
        # Store predictions
        clean_preds.append(pred_clean.detach())
        bd_preds.append(pred_attack.detach())
        cross_preds.append(pred_cross.detach())
        clean_labels.append(y_clean)
        bd_labels_all.append(y_attack)
        cross_labels.append(y_cross)
    
    # Synchronize trigger generators
    pull_model(trigger_model_prev, trigger_model)
    
    # ================================================================
    # Compute Metrics
    # ================================================================
    if not total_loss:
        return 0.0, 0.0, 0.0
    
    total_loss = float(np.average(total_loss))
    
    clean_preds = torch.cat(clean_preds, 0)
    bd_preds = torch.cat(bd_preds, 0)
    cross_preds = torch.cat(cross_preds, 0)
    clean_labels = torch.cat(clean_labels, 0)
    bd_labels_all = torch.cat(bd_labels_all, 0)
    cross_labels = torch.cat(cross_labels, 0)
    
    # Clean accuracy
    clean_predictions = torch.argmax(torch.nn.functional.softmax(clean_preds, dim=-1), dim=1).cpu().numpy()
    clean_accuracy = cal_accuracy(clean_predictions, clean_labels.flatten().cpu().numpy())
    
    # Backdoor attack success rate
    bd_predictions = torch.argmax(torch.nn.functional.softmax(bd_preds, dim=-1), dim=1).cpu().numpy()
    bd_accuracy = cal_accuracy(bd_predictions, bd_labels_all.flatten().cpu().numpy())
    
    # Cross-trigger accuracy (high = good nonreusability)
    cross_predictions = torch.argmax(torch.nn.functional.softmax(cross_preds, dim=-1), dim=1).cpu().numpy()
    cross_accuracy = cal_accuracy(cross_predictions, cross_labels.flatten().cpu().numpy())
    
    return total_loss, loss_dict, clean_accuracy, bd_accuracy


def epoch_combined_eval(
    trigger_model,
    surrogate_model,
    loader,
    args,
    freq_target: Optional[torch.Tensor] = None,
):
    """Evaluation function for combined frequency + marksman training.
    
    Returns:
        dict with 'clean_accuracy', 'attack_success_rate', 'loss_freq', 'loss_div'
    """
    trigger_model.eval()
    surrogate_model.eval()
    
    target_label = getattr(args, 'target_label', 0)
    
    if freq_target is None:
        freq_target = estimate_frequency_heatmap(surrogate_model, loader, args, max_batches=1)
    
    clean_preds, attack_preds = [], []
    clean_labels, attack_labels = [], []
    freq_losses, div_losses = [], []
    
    with torch.no_grad():
        prev_batch = None
        for batch_x, label, padding_mask in loader:
            batch_x = batch_x.float().to(args.device)
            padding_mask = padding_mask.float().to(args.device)
            label = label.to(args.device)
            
            bd_labels = torch.ones_like(label).to(args.device) * target_label
            
            # Generate triggers
            trigger, trigger_clip = trigger_model(batch_x, padding_mask, None, None, bd_labels)
            
            # Clean predictions
            pred_clean = surrogate_model(batch_x, padding_mask, None, None)
            clean_preds.append(pred_clean)
            clean_labels.append(label)
            
            # Attack predictions
            bd_input = batch_x + trigger_clip
            pred_attack = surrogate_model(bd_input, padding_mask, None, None)
            attack_preds.append(pred_attack)
            attack_labels.append(bd_labels)
            
            # Frequency loss
            loss_freq = compute_frequency_loss(trigger_clip, freq_target, args.device)
            freq_losses.append(loss_freq.item())
            
            # Diversity loss (with previous batch if available)
            if prev_batch is not None:
                prev_x, prev_trigger = prev_batch
                loss_div = diversity_loss(batch_x[:len(prev_x)], prev_x[:len(batch_x)], 
                                          trigger[:len(prev_trigger)], prev_trigger[:len(trigger)])
                div_losses.append(loss_div.item())
            prev_batch = (batch_x, trigger)
    
    results = {}
    
    clean_preds = torch.cat(clean_preds, 0)
    clean_labels = torch.cat(clean_labels, 0)
    predictions = torch.argmax(torch.nn.functional.softmax(clean_preds, dim=-1), dim=1).cpu().numpy()
    results['clean_accuracy'] = cal_accuracy(predictions, clean_labels.flatten().cpu().numpy())
    
    attack_preds = torch.cat(attack_preds, 0)
    attack_labels = torch.cat(attack_labels, 0)
    predictions = torch.argmax(torch.nn.functional.softmax(attack_preds, dim=-1), dim=1).cpu().numpy()
    results['attack_success_rate'] = cal_accuracy(predictions, attack_labels.flatten().cpu().numpy())
    
    results['loss_freq'] = np.mean(freq_losses) if freq_losses else 0.0
    results['loss_div'] = np.mean(div_losses) if div_losses else 0.0
    
    return results
