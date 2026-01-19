"""
Input-Aware Backdoor Attack with Mask Network

Full implementation matching the original paper:
"Input-Aware Dynamic Backdoor Attack" (Nguyen & Tran, NeurIPS 2020)

Key Components:
1. Pattern Generator (netG): Generates trigger patterns
2. Mask Generator (netM): Generates input-dependent spatial masks
3. Classifier (netC): Target model to be backdoored

Training Strategy:
- Stage 1: Pre-train mask network with diversity + sparsity losses
- Stage 2: Joint training of pattern generator and classifier with frozen mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.methods.utils import cal_accuracy


def mask_diversity_loss(x1, x2, mask1, mask2, epsilon=1e-3):
    """
    Diversity loss for masks: encourages different inputs to have different masks.
    
    L_div = ||x1 - x2|| / ||mask(x1) - mask(x2)||
    
    Forces the mask network to produce diverse masks for diverse inputs.
    """
    # Input distance
    input_diff = x1 - x2
    input_distances = torch.sqrt(torch.mean(input_diff ** 2, dim=(1, 2)) + epsilon)
    
    # Mask distance (use thresholded masks to match reference behavior)
    mask1_bin = (mask1 > 0.5).float()
    mask2_bin = (mask2 > 0.5).float()
    mask_diff = mask1_bin - mask2_bin
    mask_distances = torch.sqrt(torch.mean(mask_diff ** 2, dim=(1, 2)) + epsilon)
    
    # Ratio: large input difference should lead to large mask difference
    mask_distances = torch.clamp(mask_distances, min=epsilon)
    loss_div = input_distances / mask_distances
    return torch.mean(loss_div)


def mask_sparsity_loss(mask, density_target=0.032):
    """
    Sparsity loss: encourages masks to be sparse.
    
    L_norm = mean(ReLU(mask - density_target))
    
    Penalizes masks that exceed the target density.
    """
    # Only penalize values above threshold
    loss_norm = torch.mean(F.relu(mask - density_target))
    return loss_norm


def pattern_diversity_loss(x1, x2, pattern1, pattern2, epsilon=1e-3):
    """
    Diversity loss for patterns: encourages different inputs to have different patterns.
    Uses MSE-based Euclidean distances matching original implementation.
    """
    # MSE distance for inputs
    mse_inputs = torch.mean((x1 - x2) ** 2, dim=(1, 2))
    distance_inputs = torch.sqrt(mse_inputs + epsilon)
    
    # MSE distance for patterns
    mse_patterns = torch.mean((pattern1 - pattern2) ** 2, dim=(1, 2))
    distance_patterns = torch.sqrt(mse_patterns + epsilon)
    distance_patterns = torch.clamp(distance_patterns, min=epsilon)

    loss_div = distance_inputs / distance_patterns
    return torch.mean(loss_div)


def apply_masked_trigger(x, pattern, mask):
    """
    Apply trigger using input-aware mask blending.
    
    Formula: backdoor = x * (1 - mask) + pattern * mask
    
    This smoothly blends the input and pattern based on the mask.
    The mask controls WHERE the pattern is applied.
    """
    return x * (1 - mask) + pattern * mask


def create_backdoor(x, bd_labels, trigger_model, mask_model, args):
    """
    Create backdoored inputs using pattern and mask generators.
    
    Args:
        x: Clean inputs [B, T, N]
        bd_labels: Target labels for backdoor
        trigger_model: Pattern generator
        mask_model: Mask generator
        args: Configuration
    
    Returns:
        bd_inputs: Backdoored inputs
        patterns: Generated patterns
        masks: Generated masks (continuous)
        masks_binary: Binarized masks
    """
    # Generate pattern
    patterns, _ = trigger_model(x, None, None, None, bd_labels)
    
    # Generate mask
    masks = mask_model(x)
    masks_binary = mask_model.threshold(masks)
    
    # Apply masked trigger
    bd_inputs = apply_masked_trigger(x, patterns, masks_binary)
    
    return bd_inputs, patterns, masks, masks_binary


def train_mask_epoch(
    mask_model,
    loader1,
    loader2,
    args,
    opt_mask=None,
    train=True
):
    """
    Train mask generator with diversity and sparsity losses.
    
    This is Stage 1 of the input-aware attack.
    
    Args:
        mask_model: Mask generator network
        loader1, loader2: Two data loaders for diversity computation
        args: Configuration
        opt_mask: Optimizer for mask generator
        train: Training mode flag
    """
    if train:
        mask_model.train()
    else:
        mask_model.eval()
    
    total_loss = []
    loss_dict = {'L_div': [], 'L_norm': [], 'L_total': []}
    
    # Hyperparameters
    lambda_div = getattr(args, 'lambda_div', 1.0)
    lambda_norm = getattr(args, 'lambda_norm', 100.0)
    mask_density = getattr(args, 'mask_density', 0.032)
    
    for (batch_x1, _, padding_mask1), (batch_x2, _, padding_mask2) in zip(loader1, loader2):
        batch_x1 = batch_x1.float().to(args.device)
        batch_x2 = batch_x2.float().to(args.device)
        
        if train and opt_mask is not None:
            opt_mask.zero_grad()
        
        # Generate masks for both batches
        mask1 = mask_model(batch_x1)
        mask2 = mask_model(batch_x2)
        
        # Compute losses
        loss_div = mask_diversity_loss(batch_x1, batch_x2, mask1, mask2)
        loss_norm = mask_sparsity_loss(mask1, mask_density)
        
        # Total loss
        loss = lambda_div * loss_div + lambda_norm * loss_norm
        
        if train:
            loss.backward()
            if opt_mask is not None:
                # Gradient clipping for mask optimizer
                if hasattr(args, 'trigger_grad_clip') and args.trigger_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(mask_model.parameters(), args.trigger_grad_clip)
                opt_mask.step()
        
        # Logging
        total_loss.append(loss.item())
        loss_dict['L_div'].append(loss_div.item())
        loss_dict['L_norm'].append(loss_norm.item())
        loss_dict['L_total'].append(loss.item())
    
    avg_loss = np.mean(total_loss) if total_loss else 0.0
    
    return avg_loss, loss_dict


def epoch_inputaware_masking(
    bd_model,
    mask_model,
    surr_model,
    loader1,
    args,
    loader2=None,
    opt_trig=None,
    opt_class=None,
    train=True,
    mask_frozen=True
):
    """
    Input-aware backdoor attack training with mask network.
    
    This is Stage 2 of the attack (assumes mask is pre-trained).
    
    Training Strategy:
    1. Mask network is FROZEN (pre-trained)
    2. Three training modes per batch:
       - Clean: Learn correct classification
       - Attack: Learn to classify masked-triggered inputs as target
       - Cross-trigger: Learn that wrong masks/patterns don't work
    3. Pattern diversity loss: Encourage different patterns for different inputs
    
    Args:
        bd_model: Pattern generator
        mask_model: Mask generator (FROZEN)
        surr_model: Surrogate classifier
        loader1: Primary data loader
        args: Configuration
        loader2: Secondary loader for diversity
        opt_trig: Pattern generator optimizer
        opt_class: Classifier optimizer
        train: Training mode flag
        mask_frozen: Whether mask network is frozen (should be True)
    """
    total_loss = []
    all_preds, bd_preds, cross_preds = [], [], []
    trues, bds, cross_trues = [], [], []
    
    target_label = args.target_label
    num_classes = getattr(args, 'num_class', getattr(args, 'numb_class', 10))
    
    # Loss tracking
    loss_dict = {
        'L_clean': [], 'L_attack': [], 'L_cross': [],
        'L_ce': [], 'L_div_pattern': [], 'L_total': []
    }
    
    # Hyperparameters
    p_attack = getattr(args, 'p_attack', 0.4)
    p_cross = getattr(args, 'p_cross', 0.2)
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
    
    # Mask should be frozen
    if mask_frozen:
        mask_model.eval()
    
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
        else:
            indices = torch.randperm(batch_size)
            batch_x2 = batch_x[indices]
        
        # Compute batch splits
        num_attack = max(1, int(p_attack * batch_size))
        num_cross = max(1, int(p_cross * batch_size))
        num_clean = batch_size - num_attack - num_cross
        
        # Generate target labels
        bd_type = getattr(args, 'bd_type', 'all2one')
        if bd_type == 'all2all':
            bd_labels = torch.randint(0, num_classes, (batch_size,), device=args.device)
        else:
            bd_labels = torch.ones(batch_size, dtype=torch.long, device=args.device) * target_label
        
        # Zero gradients
        if train:
            if opt_class is not None:
                opt_class.zero_grad()
            if opt_trig is not None:
                opt_trig.zero_grad()
        
        # ============== GENERATE PATTERNS AND MASKS ==============
        with torch.no_grad() if mask_frozen else torch.enable_grad():
            # Generate masks (frozen)
            masks1 = mask_model(batch_x)
            masks1_binary = mask_model.threshold(masks1)
            
            masks2 = mask_model(batch_x2)
            masks2_binary = mask_model.threshold(masks2)
        
        # Generate patterns (trainable)
        # Trigger model returns (unclipped, clipped) - use unclipped for diversity loss
        patterns1_unclipped, patterns1_clipped = bd_model(batch_x, padding_mask, None, None, bd_labels)
        patterns2_unclipped, patterns2_clipped = bd_model(batch_x2, None, None, None, bd_labels)
        
        # ============== SPLIT BATCH INTO THREE MODES ==============
        # Attack mode: Apply CLIPPED pattern with its own mask
        x_attack = batch_x[:num_attack]
        y_attack = bd_labels[:num_attack]
        pattern_attack = patterns1_clipped[:num_attack]
        mask_attack = masks1_binary[:num_attack]
        bd_inputs = apply_masked_trigger(x_attack, pattern_attack, mask_attack)
        
        # Cross-trigger mode: Apply CLIPPED pattern2 with mask2 to x1 (wrong pairing)
        x_cross = batch_x[num_attack:num_attack + num_cross]
        y_cross = label[num_attack:num_attack + num_cross]
        pattern_cross = patterns2_clipped[num_attack:num_attack + num_cross]
        mask_cross = masks2_binary[num_attack:num_attack + num_cross]
        cross_inputs = apply_masked_trigger(x_cross, pattern_cross, mask_cross)
        
        # Clean mode
        x_clean = batch_x[num_attack + num_cross:]
        y_clean = label[num_attack + num_cross:]
        
        # ============== CLASSIFIER FORWARD PASS ==============
        # Concatenate all inputs for single forward pass (matches original implementation)
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
        loss_attack = args.criterion(pred_attack, y_attack.long().squeeze(-1))
        loss_cross = args.criterion(pred_cross, y_cross.long().squeeze(-1))
        loss_clean = args.criterion(pred_clean, y_clean.long().squeeze(-1))
        
        # Total classification loss
        loss_ce = loss_clean + loss_attack + loss_cross
        
        # Pattern diversity loss - use UNCLIPPED patterns (closer to original implementation)
        patterns1_for_div = patterns1_unclipped[:num_attack]
        patterns2_for_div = patterns2_unclipped[num_attack:num_attack + num_attack] if batch_x2.shape[0] >= 2*num_attack else patterns2_unclipped[:num_attack]
        inputs1_for_div = batch_x[:num_attack]
        inputs2_for_div = batch_x2[num_attack:num_attack + num_attack] if batch_x2.shape[0] >= 2*num_attack else batch_x2[:num_attack]
        
        loss_div_pattern = pattern_diversity_loss(inputs1_for_div, inputs2_for_div, patterns1_for_div, patterns2_for_div)
        loss_div_pattern = loss_div_pattern * lambda_div
        
        # ============== TOTAL LOSS ==============
        # Original implementation: CE + diversity only
        total = loss_ce + loss_div_pattern
        
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
        loss_dict['L_ce'].append(loss_ce.item())
        loss_dict['L_div_pattern'].append(loss_div_pattern.item())
        loss_dict['L_total'].append(total.item())
        
        # Store predictions
        all_preds.append(pred_clean.detach())
        bd_preds.append(pred_attack.detach())
        cross_preds.append(pred_cross.detach())
        trues.append(y_clean)
        bds.append(y_attack)
        cross_trues.append(y_cross)
    
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
