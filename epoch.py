import torch
import torch.nn as nn
import numpy as np

"""
Epoch Training Functions for Backdoor Attacks on Time Series Models

This module provides training and testing functions for both clean and backdoor scenarios:

1. clean_train / clean_test: Standard training/testing without backdoors
2. bd_train / bd_test: Train victim model on poisoned data / Test attack success rate
3. trigger_train_epoch: Train dynamic trigger generator with surrogate classifier

Standard backdoor training pipeline:
  Step 1: Use trigger_train_epoch to train a trigger generator with surrogate
  Step 2: Poison training data using trained trigger
  Step 3: Use bd_train to train victim model on poisoned data
  Step 4: Evaluate with bd_test (Attack Success Rate) and clean_test (utility)
"""

def apply_trigger(batch_x, label, args, trigger_model=None):
    """
    CLEAN modda hiçbir şey yapmaz.
    BASIC / MARKSMAN modlarda batch_x üzerine tetikleyici uygular.
    trigger_model: Bd_MLP veya TimesBA trigger network
    """
    # Clean mode -> no-op
    if args.mode == "clean":
        return batch_x, label

    # Basic backdoor (sabit patch)
    if args.mode == "basic":
        B = batch_x.size(0)
        poison_mask = torch.rand(B) < args.poisoning_ratio

        # seçilen örneklere patch ekle
        for i in range(B):
            if poison_mask[i]:
                batch_x[i, -5:, 0] += args.clip_ratio  # küçük sabit tetikleyici
                label[i] = args.target_label
        return batch_x, label

    # Trigger generator (Bd-MLP / TimesBA)
    if args.mode == "triggerNet":
        batch_x = batch_x.to(args.device)
        target_labels = torch.ones_like(label) * args.target_label
        trigger, trigger_clipped = trigger_model(batch_x, None, None, None, target_labels)
        batch_x = batch_x + trigger_clipped
        label[:] = args.target_label
        return batch_x, label

    return batch_x, label



def clean_train(model,loader,args,optimizer):
    model.train()
    total_loss = []
    preds = []
    trues = []
    for i, (batch_x, label, padding_mask) in enumerate(loader):
        model.zero_grad()
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)
        outs = model(batch_x, padding_mask, None, None)
        loss = args.criterion(outs, label.long().squeeze(-1))
        total_loss.append(loss.item())
        preds.append(outs.detach())
        trues.append(label)
        loss.backward()
        optimizer.step()
    total_loss = np.average(total_loss)
    preds = torch.cat(preds, 0)
    trues = torch.cat(trues, 0)
    probs = torch.nn.functional.softmax(preds)
    predictions = torch.argmax(probs, dim=1).cpu().numpy()
    trues = trues.flatten().cpu().numpy()
    accuracy = np.mean(predictions == trues)
    return total_loss, accuracy


def clean_test(model,loader,args):
    model.eval()
    total_loss = []
    preds = []
    trues = []
    for i, (batch_x, label, padding_mask) in enumerate(loader):
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)
        outs = model(batch_x, padding_mask, None, None)
        loss = args.criterion(outs, label.long().squeeze(-1))
        total_loss.append(loss.item())
        preds.append(outs.detach())
        trues.append(label)
    total_loss = np.average(total_loss)
    preds = torch.cat(preds, 0)
    trues = torch.cat(trues, 0)
    probs = torch.nn.functional.softmax(preds)
    predictions = torch.argmax(probs, dim=1).cpu().numpy()
    trues = trues.flatten().cpu().numpy()
    accuracy = np.mean(predictions == trues)
    return total_loss, accuracy



def bd_train(model, loader, args, optimizer, trigger_model=None):
    """Train victim model on mix of clean and poisoned samples.
    
    Standard backdoor training approach:
    - For each batch, poison a fraction (poisoning_ratio) of samples
    - Train model to classify clean samples correctly AND poisoned samples as target
    
    Args:
        model: Victim model being poisoned
        loader: Training data loader
        args: Training arguments
        optimizer: Optimizer for victim model
        trigger_model: Trigger generator (None for basic patch)
    
    Returns:
        avg_loss: Average loss over epoch
        accuracy: Training accuracy (on mixed clean+poisoned data)
    """
    model.train()
    total_loss = []
    preds = []
    trues = []

    for i, (batch_x, label, padding_mask) in enumerate(loader):
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)

        # Apply trigger to fraction of samples
        batch_x, label = apply_trigger(batch_x, label, args, trigger_model)
        
        model.zero_grad()
        outs = model(batch_x, padding_mask, None, None)
        loss = args.criterion(outs, label.long().squeeze(-1))

        total_loss.append(loss.item())
        preds.append(outs.detach())
        trues.append(label)

        loss.backward()
        optimizer.step()

    total_loss = np.average(total_loss)
    preds = torch.cat(preds, 0)
    trues = torch.cat(trues, 0)
    probs = torch.nn.functional.softmax(preds, dim=-1)
    predictions = torch.argmax(probs, dim=1).cpu().numpy()
    trues = trues.flatten().cpu().numpy()
    accuracy = np.mean(predictions == trues)
    return total_loss, accuracy


def trigger_train_epoch(trigger_model, surrogate_model, loader, args, optimizer_trigger, optimizer_surrogate):
    """Train trigger generator and surrogate classifier (one epoch).
    
    Standard approach for training dynamic triggers:
    1. Generate triggers for input samples
    2. Train surrogate to:
       - Classify clean samples correctly (maintains utility)
       - Classify triggered samples as target label (backdoor effectiveness)
    3. Update trigger generator to maximize backdoor success
    
    Args:
        trigger_model: Trigger generator network
        surrogate_model: Surrogate classifier
        loader: Training data loader
        args: Training arguments
        optimizer_trigger: Optimizer for trigger generator
        optimizer_surrogate: Optimizer for surrogate classifier
    
    Returns:
        avg_loss: Average loss
        clean_acc: Accuracy on clean samples
        bd_acc: Accuracy on backdoored samples (ASR)
    """
    trigger_model.train()
    surrogate_model.train()
    
    total_loss = []
    clean_preds = []
    bd_preds = []
    clean_labels = []
    bd_labels_list = []
    
    target_label = args.target_label
    
    for i, (batch_x, label, padding_mask) in enumerate(loader):
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)
        
        batch_size = batch_x.size(0)
        bd_labels = torch.ones_like(label) * target_label
        
        # Generate triggers
        trigger, trigger_clipped = trigger_model(batch_x, padding_mask, None, None, bd_labels)
        
        # Create triggered samples
        batch_x_bd = batch_x + trigger_clipped
        
        # Forward pass on both clean and triggered samples
        optimizer_surrogate.zero_grad()
        optimizer_trigger.zero_grad()
        
        # Clean samples - should predict correct label
        clean_out = surrogate_model(batch_x, padding_mask, None, None)
        loss_clean = args.criterion(clean_out, label.long().squeeze(-1))
        
        # Backdoor samples - should predict target label
        bd_out = surrogate_model(batch_x_bd, padding_mask, None, None)
        loss_bd = args.criterion(bd_out, bd_labels.long().squeeze(-1))
        
        # Combined loss
        loss = loss_clean + loss_bd
        
        loss.backward()
        optimizer_trigger.step()
        optimizer_surrogate.step()
        
        total_loss.append(loss.item())
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
    
    return avg_loss, clean_acc, bd_acc


def bd_test(model, loader, args, trigger_model=None,save_plot=False):
    """Test victim model's attack success rate (ASR) and clean accuracy.
    
    Applies triggers to ALL test samples and measures ASR.
    Also computes clean accuracy on original samples.
    
    Args:
        model: Victim model to test
        loader: Test data loader
        args: Test arguments
        trigger_model: Trigger generator (None for basic patch)
    
    Returns:
        avg_loss: Average loss
        clean_acc: Clean accuracy
        asr: Attack Success Rate (fraction classified as target)
    """
    model.eval()
    trigger_model.eval() if trigger_model is not None else None
    total_loss = []
    clean_preds = []
    bd_preds = []
    clean_labels = []
    bd_labels = []

    for i, (batch_x, label, padding_mask) in enumerate(loader):
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)
        original_label = label.clone()
        target_labels = torch.ones_like(label) * args.target_label

        # Apply trigger to all samples
        bd_batch_x, label = apply_trigger(batch_x, label, args, trigger_model)

        with torch.no_grad():
            outs = model(batch_x, padding_mask, None, None)
            loss_clean = args.criterion(outs, original_label.long().squeeze(-1))
            outs_bd = model(bd_batch_x, padding_mask, None, None)
            loss = args.criterion(outs_bd, target_labels.long().squeeze(-1))

        total_loss.append(loss.item() + loss_clean.item())
        clean_preds.append(outs.detach())
        bd_preds.append(outs_bd.detach())
        clean_labels.append(original_label)
        bd_labels.append(target_labels)

    if len(total_loss) == 0:
        return 0.0, 0.0, 0.0, {
            "clean_inputs": torch.empty(0),
            "triggered_inputs": torch.empty(0),
            "predictions": torch.empty(0, dtype=torch.long),
            "true_labels": torch.empty(0, dtype=torch.long),
            "sample_ids": [],
            "target_label": args.target_label,
        }

    total_loss = np.average(total_loss)
    
    clean_preds = torch.cat(clean_preds, 0)
    bd_preds = torch.cat(bd_preds, 0)
    clean_labels = torch.cat(clean_labels, 0)
    bd_labels = torch.cat(bd_labels, 0)
    
    # Clean accuracy
    clean_probs = torch.nn.functional.softmax(clean_preds, dim=-1)
    clean_predictions = torch.argmax(clean_probs, dim=1).cpu().numpy()
    clean_acc = np.mean(clean_predictions == clean_labels.flatten().cpu().numpy())
    
    # Backdoor accuracy (Attack Success Rate)
    bd_probs = torch.nn.functional.softmax(bd_preds, dim=-1)
    bd_predictions = torch.argmax(bd_probs, dim=1).cpu().numpy()
    ASR = np.mean(bd_predictions == bd_labels.flatten().cpu().numpy())
    
    return total_loss, clean_acc, ASR


def bd_test_with_samples(
    model,
    loader,
    args,
    trigger_model=None,
    max_success: int = 8,
    max_failure: int = 8,
):
    """Run bd_test while collecting sample traces for plotting.

    Samples are collected based on:
    - Success: pred == target_label and true != target_label
    - Failure: pred != target_label and true != target_label

    Returns metrics plus a sample_cases dict consumable by plot_backdoor_cases.
    """
    model.eval()
    trigger_model.eval() if trigger_model is not None else None

    total_loss = []
    clean_preds = []
    bd_preds = []
    clean_labels = []
    bd_labels = []

    sample_clean = []
    sample_triggered = []
    sample_preds = []
    sample_trues = []
    sample_ids = []
    collected_success = 0
    collected_failure = 0

    for batch_idx, (batch_x, label, padding_mask) in enumerate(loader):
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)
        original_label = label.clone()
        target_labels = torch.ones_like(label) * args.target_label

        bd_batch_x, label = apply_trigger(batch_x, label, args, trigger_model)

        with torch.no_grad():
            outs = model(batch_x, padding_mask, None, None)
            loss_clean = args.criterion(outs, original_label.long().squeeze(-1))
            outs_bd = model(bd_batch_x, padding_mask, None, None)
            loss = args.criterion(outs_bd, target_labels.long().squeeze(-1))

        total_loss.append(loss.item() + loss_clean.item())
        clean_preds.append(outs.detach())
        bd_preds.append(outs_bd.detach())
        clean_labels.append(original_label)
        bd_labels.append(target_labels)

        # Collect success/failure samples for visualization
        preds_bd_full = torch.argmax(torch.softmax(outs_bd, dim=-1), dim=1)
        non_target_mask = (original_label != target_labels).squeeze(-1)
        success_mask = (preds_bd_full == args.target_label) & non_target_mask
        failure_mask = (preds_bd_full != args.target_label) & non_target_mask

        # Helper to take from mask until quotas are met
        def _take(mask, max_needed, collected_count):
            if collected_count >= max_needed:
                return torch.empty(0, dtype=torch.long, device=mask.device), 0
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            if idx.numel() == 0:
                return torch.empty(0, dtype=torch.long, device=mask.device), 0
            take_n = min(max_needed - collected_count, idx.numel())
            return idx[:take_n], take_n

        succ_idx, succ_taken = _take(success_mask, max_success, collected_success)
        fail_idx, fail_taken = _take(failure_mask, max_failure, collected_failure)

        for i in succ_idx.cpu().tolist():
            sample_clean.append(batch_x[i].detach().cpu())
            sample_triggered.append(bd_batch_x[i].detach().cpu())
            sample_preds.append(preds_bd_full[i].detach().cpu())
            sample_trues.append(original_label[i].detach().cpu())
            sample_ids.append(f"b{batch_idx}_i{i}_succ")
        for i in fail_idx.cpu().tolist():
            sample_clean.append(batch_x[i].detach().cpu())
            sample_triggered.append(bd_batch_x[i].detach().cpu())
            sample_preds.append(preds_bd_full[i].detach().cpu())
            sample_trues.append(original_label[i].detach().cpu())
            sample_ids.append(f"b{batch_idx}_i{i}_fail")

        collected_success += succ_taken
        collected_failure += fail_taken

    total_loss = np.average(total_loss)

    clean_preds = torch.cat(clean_preds, 0)
    bd_preds = torch.cat(bd_preds, 0)
    clean_labels = torch.cat(clean_labels, 0)
    bd_labels = torch.cat(bd_labels, 0)

    clean_probs = torch.nn.functional.softmax(clean_preds, dim=-1)
    clean_predictions = torch.argmax(clean_probs, dim=1).cpu().numpy()
    clean_acc = np.mean(clean_predictions == clean_labels.flatten().cpu().numpy())

    bd_probs = torch.nn.functional.softmax(bd_preds, dim=-1)
    bd_predictions = torch.argmax(bd_probs, dim=1).cpu().numpy()
    bd_acc = np.mean(bd_predictions == bd_labels.flatten().cpu().numpy())

    sample_cases = {
        "clean_inputs": torch.stack(sample_clean, 0) if sample_clean else torch.empty(0),
        "triggered_inputs": torch.stack(sample_triggered, 0) if sample_triggered else torch.empty(0),
        "predictions": torch.stack(sample_preds, 0) if sample_preds else torch.empty(0, dtype=torch.long),
        "true_labels": torch.stack(sample_trues, 0) if sample_trues else torch.empty(0, dtype=torch.long),
        "sample_ids": sample_ids if sample_ids else None,
        "target_label": args.target_label,
    }

    return total_loss, clean_acc, bd_acc, sample_cases

# Aliases for clarity (indicates these run one full epoch)
clean_train_epoch = clean_train
clean_test_epoch = clean_test
bd_train_epoch = bd_train
bd_test_epoch = bd_test
trigger_train_epoch_alias = trigger_train_epoch  # Explicit alias for trigger training