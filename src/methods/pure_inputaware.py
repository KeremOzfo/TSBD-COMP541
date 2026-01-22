"""
Pure Input-Aware Backdoor Attack Training Module

This module implements a clean input-aware backdoor attack as described in the paper:
"Input-Aware Dynamic Backdoor Attack" (Nguyen & Tran, NeurIPS 2020)

Key Properties:
1. Diversity: Triggers vary significantly across different inputs
2. Nonreusability: A trigger generated for one input cannot activate the backdoor on another

Training Approach (Joint Optimization):
- Single forward-backward pass for both classifier and trigger generator
- Three-mode batching: clean samples, attack samples, cross-trigger samples
- Diversity loss maximizes trigger variation across inputs
- Cross-trigger loss enforces nonreusability

This is a simplified version without marksman-style alternating optimization.
"""

import torch
import numpy as np
from src.methods.utils import cal_accuracy


def diversity_loss(x1, x2, g_x1, g_x2, epsilon=1e-3):
    """
    Compute diversity enforcement loss (original implementation).
    
    L_div = ||x - x'||_2 / (||g(x) - g(x')||_2 + epsilon)
    
    Uses MSE-based Euclidean distances.
    """
    # MSE distance for inputs: mean over spatial dims, then sqrt
    mse_inputs = torch.mean((x1 - x2) ** 2, dim=(1, 2))
    distance_inputs = torch.sqrt(mse_inputs + epsilon)
    
    # MSE distance for triggers: mean over spatial dims, then sqrt
    mse_triggers = torch.mean((g_x1 - g_x2) ** 2, dim=(1, 2))
    distance_triggers = torch.sqrt(mse_triggers + epsilon)
    distance_triggers = torch.clamp(distance_triggers, min=epsilon)

    # Diversity loss: input distance / trigger distance
    loss_div = distance_inputs / distance_triggers
    return torch.mean(loss_div)


def apply_trigger(x, trigger, mask=None):
    """Apply additive trigger: B(x, t) = x + trigger"""
    if mask is not None:
        return x + trigger * mask
    return x + trigger


def epoch_pure_input_aware(
    bd_model,           # Trigger generator g
    bd_model_prev,      # Previous trigger generator (for stable training)
    surr_model,         # Surrogate classifier f
    loader1,            # Primary data loader
    args,
    loader2=None,       # Secondary data loader for diversity
    opt_trig=None,      # Optimizer for trigger generator
    opt_class=None,     # Optimizer for classifier
    train=True
):
    """
    Pure input-aware backdoor attack training (no marksman).
    
    Training Strategy:
    1. Joint optimization of classifier and trigger generator
    2. Three training modes per batch:
       - Clean: Learn correct classification
       - Attack: Learn to classify triggered inputs as target
       - Cross-trigger: Learn that wrong triggers don't work (nonreusability)
    3. Diversity loss: Encourage different triggers for different inputs
    
    Args:
        bd_model: Current trigger generator
        bd_model_prev: Frozen copy for stable classifier training
        surr_model: Surrogate classifier
        loader1: Primary data loader
        args: Configuration with:
            - p_attack: Fraction of batch for attack mode (default: 0.4)
            - p_cross: Fraction of batch for cross-trigger mode (default: 0.2)
            - lambda_div: Weight for diversity loss (default: 1.0)
            - lambda_reg: Weight for L2 regularization (default: 1e-3)
        loader2: Secondary loader for diversity computation
        opt_trig: Trigger generator optimizer
        opt_class: Classifier optimizer
        train: Training mode flag
    
    Returns:
        total_loss, loss_dict, clean_accuracy, bd_accuracy
    """
    total_loss = []
    all_preds, bd_preds, cross_preds = [], [], []
    trues, bds, cross_trues = [], [], []
    
    target_label = args.target_label
    num_classes = getattr(args, 'num_class', getattr(args, 'numb_class', 10))
    
    # Loss tracking
    loss_dict = {
        'L_clean': [], 'L_attack': [], 'L_cross': [], 
        'L_ce': [], 'L_div': [], 'L_total': []
    }
    
    # Hyperparameters
    p_attack = getattr(args, 'p_attack', 0.4)   # Attack mode probability
    p_cross = getattr(args, 'p_cross', 0.2)     # Cross-trigger probability
    lambda_div = getattr(args, 'lambda_div', 1.0)
    
    assert p_attack + p_cross < 1.0, "p_attack + p_cross must be < 1"
    
    # Handle loader2
    if loader2 is None:
        loader2 = [[None, None, None]] * len(loader1)
    
    # Set model modes
    if train:
        surr_model.train()
        bd_model.train()
    else:
        surr_model.eval()
        bd_model.eval()
    
    bd_model_prev.eval()  # Always frozen
    
    for i, ((batch_x, label, padding_mask), (batch_x2, label2, padding_mask2)) in enumerate(
        zip(loader1, loader2)
    ):
        batch_size = batch_x.shape[0]
        
        # Move to device
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device).long()
        
        if batch_x2 is not None:
            batch_x2 = batch_x2.float().to(args.device)
            padding_mask2 = padding_mask2.float().to(args.device)
        else:
            # Use shuffled version if no second loader
            indices = torch.randperm(batch_size)
            batch_x2 = batch_x[indices]
            padding_mask2 = padding_mask[indices]
        
        # Compute batch splits
        num_attack = max(1, int(p_attack * batch_size))
        num_cross = max(1, int(p_cross * batch_size))
        num_clean = batch_size - num_attack - num_cross
        
        # Generate target labels based on attack type
        bd_type = getattr(args, 'bd_type', 'all2one')
        if bd_type == 'all2all':
            bd_labels = torch.randint(0, num_classes, (batch_size,), device=args.device)
        else:  # all2one
            bd_labels = torch.ones(batch_size, dtype=torch.long, device=args.device) * target_label
        
        # Create attack mask for non-target samples only
        if getattr(args, 'attack_only_nontarget', True):
            attack_mask = (label.squeeze() != target_label).float().view(batch_size, 1, 1)
            attack_mask = attack_mask.expand(-1, batch_x.shape[1], batch_x.shape[2])
        else:
            attack_mask = torch.ones_like(batch_x)
        
        # Zero gradients
        if train:
            if opt_class is not None:
                opt_class.zero_grad()
            if opt_trig is not None:
                opt_trig.zero_grad()
        
        # ============== GENERATE TRIGGERS ==============
        # Generate triggers with current model (for both forward pass and diversity loss)
        trigger_curr, trigger_curr_clip = bd_model(batch_x, padding_mask, None, None, bd_labels)
        trigger_curr2, trigger_curr2_clip = bd_model(batch_x2, padding_mask2, None, None, bd_labels)
        
        # ============== SPLIT BATCH INTO THREE MODES ==============
        # Attack mode: B(x, g(x)) -> target class
        x_attack = batch_x[:num_attack]
        y_attack = bd_labels[:num_attack]
        trig_attack = trigger_curr_clip[:num_attack]
        mask_attack = attack_mask[:num_attack]
        
        # Cross-trigger mode: B(x, g(x')) -> original class (nonreusability)
        x_cross = batch_x[num_attack:num_attack + num_cross]
        y_cross = label[num_attack:num_attack + num_cross]  # Original label!
        trig_cross = trigger_curr2_clip[num_attack:num_attack + num_cross]
        mask_cross = attack_mask[num_attack:num_attack + num_cross]
        
        # Clean mode: x -> original class
        x_clean = batch_x[num_attack + num_cross:]
        y_clean = label[num_attack + num_cross:]
        
        # ============== CLASSIFIER FORWARD PASS ==============
        # Original implementation: concatenate all inputs for single forward pass
        # Attack samples with triggers
        bd_inputs = apply_trigger(x_attack, trig_attack, mask_attack)
        
        # Cross-trigger samples (trigger from different input)
        cross_inputs = apply_trigger(x_cross, trig_cross, mask_cross)
        
        # Concatenate: [bd_inputs, cross_inputs, clean_inputs]
        total_inputs = torch.cat([bd_inputs, cross_inputs, x_clean], dim=0)
        total_padding_mask = torch.cat([padding_mask[:num_attack], 
                                         padding_mask[num_attack:num_attack + num_cross],
                                         padding_mask[num_attack + num_cross:]], dim=0)
        
        # Single forward pass
        total_preds = surr_model(total_inputs, total_padding_mask, None, None)
        
        # Split predictions back
        pred_attack = total_preds[:num_attack]
        pred_cross = total_preds[num_attack:num_attack + num_cross]
        pred_clean = total_preds[num_attack + num_cross:]
        
        # ============== COMPUTE LOSSES ==============
        # Classification losses
        loss_attack = args.criterion(pred_attack, y_attack.long().squeeze(-1))
        loss_cross = args.criterion(pred_cross, y_cross.long().squeeze(-1))
        loss_clean = args.criterion(pred_clean, y_clean.long().squeeze(-1))
        
        # Total classification loss
        loss_ce = loss_clean + loss_attack + loss_cross
        
        # Diversity loss: compare UNCLIPPED patterns (closer to original implementation)
        # Original: inputs1[:num_bd] vs inputs2[num_bd:num_bd+num_bd]
        # Use unclipped triggers to avoid tiny denominators after clipping
        patterns1_for_div = trigger_curr[:num_attack]
        patterns2_for_div = trigger_curr2[num_attack:num_attack + num_attack] if batch_x2.shape[0] >= 2*num_attack else trigger_curr2[:num_attack]
        inputs1_for_div = batch_x[:num_attack]
        inputs2_for_div = batch_x2[num_attack:num_attack + num_attack] if batch_x2.shape[0] >= 2*num_attack else batch_x2[:num_attack]
        
        loss_div = diversity_loss(inputs1_for_div, inputs2_for_div, patterns1_for_div, patterns2_for_div)
        loss_div = loss_div * lambda_div
        
        # ============== TOTAL LOSS ==============
        # Original implementation: just CE + diversity (no regularization)
        total = loss_ce + loss_div
        
        # ============== BACKWARD PASS ==============
        if train:
            total.backward()
            
            # Gradient clipping
            if hasattr(args, 'trigger_grad_clip') and args.trigger_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(bd_model.parameters(), args.trigger_grad_clip)
            if hasattr(args, 'surrogate_grad_clip') and args.surrogate_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(surr_model.parameters(), args.surrogate_grad_clip)
            
            if opt_class is not None:
                opt_class.step()
            if opt_trig is not None:
                opt_trig.step()
        
        # ============== LOGGING ==============
        total_loss.append(total.item())
        loss_dict['L_clean'].append(loss_clean.item())
        loss_dict['L_attack'].append(loss_attack.item())
        loss_dict['L_cross'].append(loss_cross.item())
        loss_dict['L_div'].append(loss_div.item())
        loss_dict['L_ce'].append(loss_ce.item())
        loss_dict['L_total'].append(total.item())
        
        # Store predictions
        all_preds.append(pred_clean.detach())
        bd_preds.append(pred_attack.detach())
        cross_preds.append(pred_cross.detach())
        trues.append(y_clean)
        bds.append(y_attack)
        cross_trues.append(y_cross)
    
    # ============== SYNC TRIGGER GENERATORS ==============
    # Copy current model to previous for next epoch
    if train:
        bd_model_prev.load_state_dict(bd_model.state_dict())
    
    # ============== COMPUTE METRICS ==============
    if not total_loss:
        return 0.0, loss_dict, 0.0, 0.0
    
    avg_loss = np.mean(total_loss)
    
    # Clean accuracy
    all_preds = torch.cat(all_preds, 0)
    trues = torch.cat(trues, 0)
    clean_predictions = torch.argmax(torch.softmax(all_preds, dim=-1), dim=1).cpu().numpy()
    clean_accuracy = cal_accuracy(clean_predictions, trues.flatten().cpu().numpy())
    
    # Backdoor accuracy (ASR)
    bd_preds = torch.cat(bd_preds, 0)
    bds = torch.cat(bds, 0)
    bd_predictions = torch.argmax(torch.softmax(bd_preds, dim=-1), dim=1).cpu().numpy()
    bd_accuracy = cal_accuracy(bd_predictions, bds.flatten().cpu().numpy())
    
    return avg_loss, loss_dict, clean_accuracy, bd_accuracy
