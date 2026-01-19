"""
Input-Aware Backdoor Attack Training Module

This module implements the strictly input-aware backdoor attack as described in the diversity.tex paper.
The attack satisfies:
1. Diversity: triggers vary significantly across different inputs
2. Nonreusability: a trigger generated for one input cannot activate the backdoor on another input

The training uses three modes:
- Clean mode: model learns to classify clean inputs correctly
- Attack mode: model learns to classify triggered inputs as target class
- Cross-trigger mode: model learns to classify inputs with "wrong" triggers (from other samples) correctly

"""

import math
import torch
import numpy as np
import torch.nn as nn
from typing import TYPE_CHECKING
from src.methods.utils import cal_accuracy

if TYPE_CHECKING:  # pragma: no cover - static checking only
    from utils.model_ops import pull_model as pull_model  # type: ignore
else:
    # pull_model utility may not exist in this repo; provide a fallback copier
    try:  # pragma: no cover - optional dependency
        from utils.model_ops import pull_model  # type: ignore
    except Exception:  # pragma: no cover - fallback copy
        def pull_model(dst, src):
            dst.load_state_dict(src.state_dict())



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
        epsilon: Small constant for numerical stability
    
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


def apply_trigger(x, trigger, mask=None):
    """
    Apply trigger to input: B(x, t) = x + trigger (additive model)
    For the paper's formulation: B(x,t) = x ⊙ (1-m) + p ⊙ m
    Here we use additive triggers where the trigger already incorporates the mask.
    
    Args:
        x: Clean input
        trigger: Generated trigger (already clipped)
        mask: Optional mask for selective triggering
    
    Returns:
        Triggered input
    """
    if mask is not None:
        return x + trigger * mask
    return x + trigger


def epoch_marksman_input_aware(
    bd_model,           # Current trigger generator g
    bd_model_prev,      # Previous trigger generator (for stable classifier training)
    surr_model,         # Surrogate classifier f
    loader1,            # Primary data loader
    args,
    loader2=None,       # Secondary data loader for diversity/cross-trigger
    opt_trig=None,      # Optimizer for trigger generator
    opt_class=None,     # Optimizer for classifier
    train=True
):
    """
    Implements the strictly input-aware backdoor attack training (Algorithm 1 from diversity.tex).
    
    This function trains both the classifier f and trigger generator g with:
    1. Classification loss from three modes (clean, attack, cross-trigger)
    2. Diversity loss to ensure trigger variety
    
    The key insight for NONREUSABILITY:
    - In cross-trigger mode, we apply trigger g(x') to input x
    - The model should predict the ORIGINAL label y (not target c)
    - This enforces that triggers are input-specific and cannot be reused
    
    Args:
        bd_model: Trigger generator network g (being trained)
        bd_model_prev: Frozen copy of trigger generator (for stable training)
        surr_model: Classifier network f
        loader1: Primary data loader
        args: Arguments containing:
            - target_label: Target class c for backdoor
            - p_attack (rho_b): Probability of attack mode
            - p_cross (rho_c): Probability of cross-trigger mode
            - div_reg (lambda_div): Weight for diversity loss
            - attack_only_nontarget: Whether to only attack non-target samples
        loader2: Secondary data loader for cross-trigger samples
        opt_trig: Optimizer for trigger generator
        opt_class: Optimizer for classifier
        train: Whether in training mode
    
    Returns:
        total_loss: Average total loss
        loss_dict: Dictionary with breakdown of losses
        accuracy: Clean accuracy
        bd_accuracy: Backdoor attack success rate
    """
    total_loss = []
    all_preds = []
    bd_preds = []
    cross_preds = []
    trues = []
    bds = []
    cross_trues = []
    
    bd_label = args.target_label
    loss_dict = {'CE_clean': [], 'CE_bd': [], 'CE_cross': [], 'div': [], 'reg': []}
    
    # Probabilities for each mode (from paper)
    rho_b = getattr(args, 'p_attack', 0.5)      # backdoor probability ρ_b
    rho_c = getattr(args, 'p_cross', 0.1)       # cross-trigger probability ρ_c
    lambda_div = getattr(args, 'div_reg', 1.0)  # diversity loss weight λ_div
    
    # Marksman-style parameters for alternating optimization
    alpha = getattr(args, 'marksman_alpha', 1.0)        # weight for backdoor loss in classifier training
    beta = getattr(args, 'marksman_beta', 0.0)          # trigger magnitude penalty
    update_T = getattr(args, 'marksman_update_T', 1)    # trigger update interval (1 = every step)
    
    assert rho_b + rho_c < 1.0, "ρ_b + ρ_c must be less than 1"
    
    # Ensure loader2 is available for cross-trigger mode
    if loader2 is None:
        loader2 = [[None, None, None]] * len(loader1)
    
    bd_model_prev.eval()  # Frozen trigger generator for classifier training
    
    if train:
        surr_model.train()
        bd_model.train()
    else:
        surr_model.eval()
        bd_model.eval()
    
    for i, (batch_x, label, padding_mask), (batch_x2, label2, padding_mask2) in zip(
        range(len(loader1)), loader1, loader2
    ):
        batch_size = batch_x.shape[0]
        
        # Move data to device
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device).long()
        
        if batch_x2 is not None:
            batch_x2 = batch_x2.float().to(args.device)
            padding_mask2 = padding_mask2.float().to(args.device)
            label2 = label2.to(args.device).long()
        else:
            # If no second loader, use shuffled version of first batch
            indices = torch.randperm(batch_size)
            batch_x2 = batch_x[indices]
            padding_mask2 = padding_mask[indices]
            label2 = label[indices]
        
        # Generate backdoor labels (target class c)
        if args.bd_type == 'all2all':
            bd_labels = torch.randint(0, args.num_class, (batch_size,), dtype=torch.long, device=args.device)
        elif args.bd_type == 'all2one':
            bd_labels = (torch.ones_like(label) * bd_label).long().to(args.device)
        else:
            raise ValueError('bd_type should be all2all or all2one')
        
        # Compute number of samples for each mode
        num_bd = int(rho_b * batch_size)
        num_cross = int(rho_c * batch_size)
        num_clean = batch_size - num_bd - num_cross
        
        # Ensure we have at least one sample in each mode
        num_bd = max(1, num_bd)
        num_cross = max(1, min(num_cross, batch_size - num_bd - 1))
        num_clean = batch_size - num_bd - num_cross
        
        # Create mask for attacking only non-target samples if specified
        if args.attack_only_nontarget:
            attack_mask = (label.squeeze() != bd_label).float().to(args.device)
        else:
            attack_mask = torch.ones(batch_size, dtype=torch.float32, device=args.device)
        
        # Expand mask to match batch_x dimensions [B, T, C]
        if len(batch_x.shape) == 3:
            attack_mask = attack_mask.view(batch_size, 1, 1).expand(-1, batch_x.shape[1], batch_x.shape[2])
        
        # ============== PHASE 1: Train Classifier with Frozen Trigger Generator ==============
        bd_model.zero_grad()
        surr_model.zero_grad()
        
        with torch.no_grad():
            # Generate triggers with frozen model
            trigger_t, trigger_t_clip = bd_model_prev(batch_x, padding_mask, None, None, bd_labels)
            trigger_t_prime, trigger_t_prime_clip = bd_model_prev(batch_x2, padding_mask2, None, None, bd_labels)
        
        # Split batch into three modes
        # Attack mode samples (indices 0 to num_bd)
        x_attack = batch_x[:num_bd]
        y_attack = bd_labels[:num_bd]  # Target label c
        trigger_attack = trigger_t_clip[:num_bd]
        mask_attack = attack_mask[:num_bd]
        
        # Cross-trigger mode samples (indices num_bd to num_bd+num_cross)
        x_cross = batch_x[num_bd:num_bd + num_cross]
        y_cross = label[num_bd:num_bd + num_cross]  # Original label y (NOT target!)
        trigger_cross = trigger_t_prime_clip[num_bd:num_bd + num_cross]  # Trigger from x'
        mask_cross = attack_mask[num_bd:num_bd + num_cross]
        
        # Clean mode samples (remaining)
        x_clean = batch_x[num_bd + num_cross:]
        y_clean = label[num_bd + num_cross:]
        
        # Compute predictions for each mode
        # Attack mode: B(x, g(x)) -> should predict c
        bd_inputs = apply_trigger(x_attack, trigger_attack, mask_attack)
        pred_attack = surr_model(bd_inputs, padding_mask[:num_bd], None, None)
        
        # Cross-trigger mode: B(x, g(x')) -> should predict y (NONREUSABILITY!)
        cross_inputs = apply_trigger(x_cross, trigger_cross, mask_cross)
        pred_cross = surr_model(cross_inputs, padding_mask[num_bd:num_bd + num_cross], None, None)
        
        # Clean mode: x -> should predict y
        pred_clean = surr_model(x_clean, padding_mask[num_bd + num_cross:], None, None)
        
        # Compute classification losses
        loss_ce_attack = args.criterion(pred_attack, y_attack.long().squeeze(-1))
        loss_ce_cross = args.criterion(pred_cross, y_cross.long().squeeze(-1))  # Original label!
        loss_ce_clean = args.criterion(pred_clean, y_clean.long().squeeze(-1))
        
        # Total classification loss (alpha weights the backdoor losses)
        loss_class = loss_ce_clean + alpha * (loss_ce_attack + loss_ce_cross)
        
        if opt_class is not None and train:
            loss_class.backward()
            
            # Gradient clipping
            if hasattr(args, 'surrogate_grad_clip') and args.surrogate_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(surr_model.parameters(), args.surrogate_grad_clip)
            
            opt_class.step()
        
        # ============== PHASE 2: Train Trigger Generator ==============
        # Generate triggers with active model (always needed for logging)
        trigger_t, trigger_t_clip = bd_model(batch_x, padding_mask, None, None, bd_labels)
        trigger_t_prime, trigger_t_prime_clip = bd_model(batch_x2, padding_mask2, None, None, bd_labels)
        
        # ============== Diversity Loss (Equation from paper) ==============
        # L_div = ||x - x'|| / ||g(x) - g(x')||
        loss_div_raw = diversity_loss(batch_x, batch_x2, trigger_t, trigger_t_prime)
        loss_div = loss_div_raw * lambda_div
        
        # ============== Backdoor Loss for Trigger Generator ==============
        # The trigger should make classifier predict target class
        bd_inputs_trig = apply_trigger(batch_x[:num_bd], trigger_t_clip[:num_bd], mask_attack)
        pred_bd_trig = surr_model(bd_inputs_trig, padding_mask[:num_bd], None, None)
        loss_bd_trig = args.criterion(pred_bd_trig, bd_labels[:num_bd].long().squeeze(-1))
        
        # ============== Cross-Trigger Loss for Nonreusability ==============
        # When using trigger from x' on x, model should predict original label y
        # This enforces that triggers cannot be reused across inputs
        cross_inputs_trig = apply_trigger(
            batch_x[num_bd:num_bd + num_cross],
            trigger_t_prime_clip[num_bd:num_bd + num_cross],
            mask_cross
        )
        pred_cross_trig = surr_model(cross_inputs_trig, padding_mask[num_bd:num_bd + num_cross], None, None)
        loss_cross_trig = args.criterion(pred_cross_trig, label[num_bd:num_bd + num_cross].long().squeeze(-1))
        
        
        # Total trigger loss with beta penalty (marksman-style)
        # L_total = L_bd + L_cross + λ_div * L_div - β * ||g||²
        # Note: beta acts as trigger magnitude penalty (negative regularization)
        loss_trig = loss_bd_trig + loss_cross_trig + loss_div - beta * torch.mean(trigger_t ** 2)
        
        # Only update trigger every update_T steps (marksman-style)
        if train and opt_trig is not None and (i % update_T == 0):
            surr_model.eval()  # Freeze classifier for trigger training
            bd_model.zero_grad()
            loss_trig.backward()
            
            # Gradient clipping
            if hasattr(args, 'trigger_grad_clip') and args.trigger_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(bd_model.parameters(), args.trigger_grad_clip)
            
            opt_trig.step()
        
        # ============== Logging ==============
        total_loss.append((loss_class.item() + loss_trig.item()))
        loss_dict['CE_clean'].append(loss_ce_clean.item())
        loss_dict['CE_bd'].append(loss_ce_attack.item())
        loss_dict['CE_cross'].append(loss_ce_cross.item())
        loss_dict['div'].append(loss_div.item())
        loss_dict['reg'].append((beta * torch.mean(trigger_t ** 2)).item())  # Track beta penalty
        
        # Store predictions for accuracy calculation
        all_preds.append(pred_clean.detach())
        bd_preds.append(pred_attack.detach())
        cross_preds.append(pred_cross.detach())
        trues.append(y_clean)
        bds.append(y_attack)
        cross_trues.append(y_cross)
    
    # Synchronize trigger generators
    pull_model(bd_model_prev, bd_model)
    
    # Compute metrics
    total_loss = np.average(total_loss)
    
    all_preds = torch.cat(all_preds, 0)
    bd_preds = torch.cat(bd_preds, 0)
    cross_preds = torch.cat(cross_preds, 0)
    trues = torch.cat(trues, 0)
    bd_labels_all = torch.cat(bds, 0)
    cross_trues = torch.cat(cross_trues, 0)
    
    # Clean accuracy
    predictions = torch.argmax(torch.nn.functional.softmax(all_preds, dim=-1), dim=1).cpu().numpy()
    trues_np = trues.flatten().cpu().numpy()
    accuracy = cal_accuracy(predictions, trues_np)
    
    # Backdoor attack success rate
    bd_predictions = torch.argmax(torch.nn.functional.softmax(bd_preds, dim=-1), dim=1).cpu().numpy()
    bd_accuracy = cal_accuracy(bd_predictions, bd_labels_all.flatten().cpu().numpy())
    
    # Cross-trigger accuracy (should be high = nonreusability)
    cross_predictions = torch.argmax(torch.nn.functional.softmax(cross_preds, dim=-1), dim=1).cpu().numpy()
    cross_accuracy = cal_accuracy(cross_predictions, cross_trues.flatten().cpu().numpy())
    
    # Add cross accuracy to loss dict for monitoring
    loss_dict['cross_acc'] = [cross_accuracy]
    
    return total_loss, loss_dict, accuracy, bd_accuracy


def epoch_marksman_input_aware_eval(
    bd_model,
    surr_model,
    loader1,
    args,
    loader2=None,
    mode='all'  # 'clean', 'attack', 'cross', or 'all'
):
    """
    Evaluation function for the input-aware backdoor attack.
    
    This function evaluates:
    1. Clean accuracy (CA): accuracy on clean inputs
    2. Attack success rate (ASR): accuracy of backdoor attack
    3. Cross-trigger accuracy (CTA): accuracy when using wrong triggers (nonreusability)
    
    High CTA means the attack satisfies the nonreusability criterion.
    
    Args:
        bd_model: Trigger generator
        surr_model: Classifier
        loader1: Primary data loader
        args: Arguments
        loader2: Secondary data loader for cross-trigger evaluation
        mode: Which metrics to compute
    
    Returns:
        Dictionary with evaluation metrics
    """
    bd_model.eval()
    surr_model.eval()
    
    clean_preds = []
    attack_preds = []
    cross_preds = []
    clean_labels = []
    attack_labels = []
    cross_labels = []
    
    bd_label = args.target_label
    
    if loader2 is None:
        loader2 = [[None, None, None]] * len(loader1)
    
    with torch.no_grad():
        for (batch_x, label, padding_mask), (batch_x2, label2, padding_mask2) in zip(loader1, loader2):
            batch_x = batch_x.float().to(args.device)
            padding_mask = padding_mask.float().to(args.device)
            label = label.to(args.device)
            
            if batch_x2 is not None:
                batch_x2 = batch_x2.float().to(args.device)
                padding_mask2 = padding_mask2.float().to(args.device)
            else:
                indices = torch.randperm(batch_x.shape[0])
                batch_x2 = batch_x[indices]
                padding_mask2 = padding_mask[indices]
            
            bd_labels = torch.ones_like(label).to(args.device) * bd_label
            
            # Generate triggers
            _, trigger_clip = bd_model(batch_x, padding_mask, None, None, bd_labels)
            _, trigger_clip2 = bd_model(batch_x2, padding_mask2, None, None, bd_labels)
            
            if mode in ['clean', 'all']:
                pred_clean = surr_model(batch_x, padding_mask, None, None)
                clean_preds.append(pred_clean)
                clean_labels.append(label)
            
            if mode in ['attack', 'all']:
                bd_input = batch_x + trigger_clip
                pred_attack = surr_model(bd_input, padding_mask, None, None)
                attack_preds.append(pred_attack)
                attack_labels.append(bd_labels)
            
            if mode in ['cross', 'all']:
                # Cross-trigger: apply trigger from x2 to x1
                cross_input = batch_x + trigger_clip2
                pred_cross = surr_model(cross_input, padding_mask, None, None)
                cross_preds.append(pred_cross)
                cross_labels.append(label)  # Should predict original label!
    
    results = {}
    
    if clean_preds:
        clean_preds = torch.cat(clean_preds, 0)
        clean_labels = torch.cat(clean_labels, 0)
        predictions = torch.argmax(torch.nn.functional.softmax(clean_preds, dim=-1), dim=1).cpu().numpy()
        results['clean_accuracy'] = cal_accuracy(predictions, clean_labels.flatten().cpu().numpy())
    
    if attack_preds:
        attack_preds = torch.cat(attack_preds, 0)
        attack_labels = torch.cat(attack_labels, 0)
        predictions = torch.argmax(torch.nn.functional.softmax(attack_preds, dim=-1), dim=1).cpu().numpy()
        results['attack_success_rate'] = cal_accuracy(predictions, attack_labels.flatten().cpu().numpy())
    
    if cross_preds:
        cross_preds = torch.cat(cross_preds, 0)
        cross_labels = torch.cat(cross_labels, 0)
        predictions = torch.argmax(torch.nn.functional.softmax(cross_preds, dim=-1), dim=1).cpu().numpy()
        # High cross_accuracy = good nonreusability (triggers don't transfer)
        results['cross_trigger_accuracy'] = cal_accuracy(predictions, cross_labels.flatten().cpu().numpy())
        # Low attack rate with wrong trigger = good nonreusability
        bd_labels_cross = torch.ones_like(cross_labels).to(args.device) * bd_label
        results['cross_trigger_attack_rate'] = cal_accuracy(
            predictions, bd_labels_cross.flatten().cpu().numpy()
        )
    
    return results


