import torch
import numpy as np

from .utils import cal_accuracy


def epoch_vanilla_trigger_training(
    trigger_model,
    surrogate_model,
    loader,
    args,
    optimizer_trigger=None,
    optimizer_surrogate=None,
    train: bool = True,
):
    """Train trigger network and surrogate network for one epoch.
    
    Adversarial loss:
    $ L_{adv}= \sum_{X \in \mathcal{D}_{i}} L_{CE}(F_{cl}(F_{tr}(X; \boldsymbol{\theta}_{tr}, \delta); \boldsymbol{\theta}_{cl}), \tilde{y})$ 
    Clean loss:
    $L_{clean}= \sum_{X \in \mathcal{D}_{i}} L_{CE}(F_{cl}(X;\boldsymbol{\theta}_{cl}),y)$
    Objective is to minimize $ L_{adv} + L_{clean} $

    Args:
        trigger_model: Trigger generator network
        surrogate_model: Surrogate classifier
        loader: Training data loader
        args: Training arguments (includes bd_type, target_label, num_class)
        optimizer_trigger: Optimizer for trigger generator (optional, only used if train=True)
        optimizer_surrogate: Optimizer for surrogate classifier (optional, only used if train=True)
        train: Whether to train (True) or evaluate (False)
    
    Returns:
        avg_loss: Average loss
        clean_acc: Accuracy on clean samples
        bd_acc: Accuracy on backdoored samples (ASR)
    """
    if train:
        trigger_model.train()
        surrogate_model.train()
    else:
        trigger_model.eval()
        surrogate_model.eval()
    
    total_loss = []
    clean_preds = []
    bd_preds = []
    clean_labels = []
    bd_labels_list = []
    
    target_label = args.target_label
    bd_type = getattr(args, 'bd_type', 'all2one')
    num_classes = getattr(args, 'num_class', getattr(args, 'numb_class', None))
    
    # Initialize loss tracking dict
    loss_dict = {
        'L_clean': [],
        'L_bd': [],
        'L_total': []
    }
    
    for i, (batch_x, label, padding_mask) in enumerate(loader):
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)
        
        batch_size = batch_x.size(0)
        
        # Generate backdoor labels based on attack type
        if bd_type == 'all2all' and num_classes is not None:
            bd_labels = torch.randint(0, num_classes, label.shape, device=args.device)
        else:  # all2one
            bd_labels = torch.ones_like(label) * target_label
        
        # Generate triggers
        trigger, trigger_clipped = trigger_model(batch_x, padding_mask, None, None, bd_labels)
        
        # Create triggered samples
        batch_x_bd = batch_x + trigger_clipped
        
        # Forward pass on both clean and triggered samples
        if train and optimizer_surrogate is not None:
            optimizer_surrogate.zero_grad()
        if train and optimizer_trigger is not None:
            optimizer_trigger.zero_grad()
        
        # Clean samples - should predict correct label
        clean_out = surrogate_model(batch_x, padding_mask, None, None)
        loss_clean = args.criterion(clean_out, label.long().squeeze(-1))
        
        # Backdoor samples - should predict target label
        bd_out = surrogate_model(batch_x_bd, padding_mask, None, None)
        loss_bd = args.criterion(bd_out, bd_labels.long().squeeze(-1))
        
        # Combined loss
        loss = loss_clean + loss_bd
        
        if train and optimizer_trigger is not None and optimizer_surrogate is not None:
            loss.backward()
            
            # Gradient clipping
            if hasattr(args, 'trigger_grad_clip') and args.trigger_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trigger_model.parameters(), args.trigger_grad_clip)
            if hasattr(args, 'surrogate_grad_clip') and args.surrogate_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), args.surrogate_grad_clip)
            
            optimizer_trigger.step()
            optimizer_surrogate.step()
        
        total_loss.append(loss.item())
        loss_dict['L_clean'].append(loss_clean.item())
        loss_dict['L_bd'].append(loss_bd.item())
        loss_dict['L_total'].append(loss.item())
        clean_preds.append(clean_out.detach())
        bd_preds.append(bd_out.detach())
        clean_labels.append(label)
        bd_labels_list.append(bd_labels)
    
    # Compute metrics
    avg_loss = np.average(total_loss)
    
    clean_preds = torch.cat(clean_preds, 0)
    bd_preds = torch.cat(bd_preds, 0)
    clean_labels = torch.cat(clean_labels, 0)
    bd_labels_all = torch.cat(bd_labels_list, 0)
    
    # Clean accuracy
    clean_probs = torch.nn.functional.softmax(clean_preds, dim=-1)
    clean_predictions = torch.argmax(clean_probs, dim=1).cpu().numpy()
    clean_acc = np.mean(clean_predictions == clean_labels.flatten().cpu().numpy())
    
    # Backdoor accuracy (Attack Success Rate)
    bd_probs = torch.nn.functional.softmax(bd_preds, dim=-1)
    bd_predictions = torch.argmax(bd_probs, dim=1).cpu().numpy()
    bd_acc = np.mean(bd_predictions == bd_labels_all.flatten().cpu().numpy())
    
    return avg_loss, loss_dict, clean_acc, bd_acc
