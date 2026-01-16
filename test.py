"""
Testing Module for TimeSeries Backdoor Attack

This module is used for testing already-trained trigger models.
It loads a pre-trained trigger model, poisons training data, trains a fresh
victim model on the poisoned data, and evaluates the attack success rate.
"""

import torch
import numpy as np
import functools
import random
from torch.utils.data import DataLoader, Subset

from data_provider.data_factory import data_provider
from data_provider.uea import collate_fn
from utils.exp_logging import log_all, log_final_test_epoch  # log_final_test_epoch may lazy-import from this module
from utils.helper_train_test import get_data, reconfigure_model_for_data, target_model_dict, trigger_model_dict
from utils.tools import create_optimizer


def apply_trigger(batch_x, label, args, trigger_model=None, mask_model=None):
    """Apply trigger to a batch of samples.
    
    Args:
        batch_x: Batch of clean inputs (B, T, C)
        label: Original labels
        args: Arguments with target_label and clip_ratio
        trigger_model: Trained trigger model (None for basic patch trigger)
        mask_model: Mask generator (for inputaware_masking method)
    
    Returns:
        triggered_batch: Batch with triggers applied
        bd_labels: Backdoor target labels
    """
    batch_size = batch_x.shape[0]
    target_labels = (torch.ones_like(label) * args.target_label).long().to(args.device)
    
    if trigger_model is not None:
        trigger_model.eval()
        with torch.no_grad():
            pattern, trigger_clip = trigger_model(batch_x, None, None, None, target_labels)
            
            # Apply mask if available
            if mask_model is not None:
                mask = mask_model(batch_x)
                mask_binary = mask_model.threshold(mask)
                # Blending: x * (1 - mask) + pattern * mask
                triggered_batch = batch_x * (1 - mask_binary) + trigger_clip * mask_binary
            else:
                # Simple additive
                triggered_batch = batch_x + trigger_clip
    else:
        # Basic patch trigger
        triggered_batch = batch_x.clone()
        clip_ratio = getattr(args, 'clip_ratio', 0.1)
        triggered_batch[:, -5:, 0] += clip_ratio
    
    return triggered_batch, target_labels


def poison_data_with_trigger(trigger_model, train_data, args):
    """Apply triggers to training data to create poisoned samples.
    
    Args:
        trigger_model: Trained trigger generator
        train_data: Original training dataset
        args: Arguments containing poisoning ratio and target label
    
    Returns:
        poisoned_dataset: Dataset with poisoned samples
        poison_indices: List of poisoned sample indices
    """
    print(f"Poisoning {args.poisoning_ratio*100:.1f}% of training data...")
    
    total_samples = len(train_data)
    num_poison = int(total_samples * args.poisoning_ratio)
    
    if num_poison == 0:
        print("No samples to poison (ratio too low)")
        return train_data, []
    
    poison_indices = random.sample(range(total_samples), num_poison)
    
    # Make DataFrames writable
    train_data.feature_df = train_data.feature_df.copy()
    train_data.labels_df = train_data.labels_df.copy()
    
    if trigger_model is not None:
        trigger_model.eval()
    
    with torch.no_grad():
        for idx in poison_indices:
            sample_data = train_data[idx]
            
            if isinstance(sample_data, tuple):
                x, y = sample_data[0], sample_data[1]
            else:
                x = sample_data
            
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            
            x = x.unsqueeze(0).to(args.device).float()
            
            if trigger_model is not None:
                target_label = torch.tensor([args.target_label]).to(args.device)
                _, trigger_clip = trigger_model(x, None, None, None, target_label)
                x_poisoned = x + trigger_clip
            else:
                x_poisoned = x.clone()
                x_poisoned[0, -5:, 0] += getattr(args, 'clip_ratio', 0.1)
            
            x_poisoned = x_poisoned.squeeze(0).cpu()
            train_data.feature_df.loc[train_data.all_IDs[idx]] = x_poisoned.numpy()
            train_data.labels_df.loc[train_data.all_IDs[idx]] = args.target_label
    
    print(f"Poisoned {num_poison} samples")
    return train_data, poison_indices


def clean_train(model, loader, args, optimizer):
    """Train model on clean data for one epoch.
    
    Args:
        model: Model to train
        loader: Data loader
        args: Training arguments
        optimizer: Optimizer
    
    Returns:
        avg_loss: Average loss
        accuracy: Clean accuracy
    """
    model.train()
    total_loss = []
    preds = []
    trues = []
    
    for batch_x, label, padding_mask in loader:
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)
        
        optimizer.zero_grad()
        outs = model(batch_x, padding_mask, None, None)
        loss = args.criterion(outs, label.long().squeeze(-1))
        loss.backward()
        optimizer.step()
        
        total_loss.append(loss.item())
        preds.append(outs.detach())
        trues.append(label)
    
    avg_loss = np.average(total_loss)
    preds = torch.cat(preds, 0)
    trues = torch.cat(trues, 0)
    probs = torch.nn.functional.softmax(preds, dim=-1)
    predictions = torch.argmax(probs, dim=1).cpu().numpy()
    accuracy = np.mean(predictions == trues.flatten().cpu().numpy())
    
    return avg_loss, accuracy


def test_pretrained_trigger(args, trigger_model_path):
    """Test a pre-trained trigger model by training a fresh victim model.
    
    This function:
    1. Loads a pre-trained trigger model
    2. Poisons training data using the trigger
    3. Trains a fresh victim model on poisoned data
    4. Evaluates attack success rate
    
    Args:
        args: Test arguments from args_parser()
        trigger_model_path: Path to the saved trigger model checkpoint
        
    Returns:
        dict: Results with CA and ASR over epochs
    """
    print("\n" + "="*60)
    print("TESTING PRE-TRAINED TRIGGER MODEL")
    print("="*60 + "\n")
    
    # Load data
    train_data, train_loader = get_data(args, 'train')
    test_data, test_loader = get_data(args, 'test')
    
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    # Load pre-trained trigger model
    print(f"Loading trigger model from: {trigger_model_path}")
    trigger_model = trigger_model_dict[args.Tmodel](args).float().to(args.device)
    trigger_model.load_state_dict(torch.load(trigger_model_path, map_location=args.device))
    trigger_model.eval()
    print("Trigger model loaded successfully!")
    
    # Create fresh victim model
    print(f"Creating fresh {args.model} victim model...")
    victim_model = reconfigure_model_for_data(args, train_data, test_data)
    
    # Poison training data
    poisoned_data, poison_indices = poison_data_with_trigger(trigger_model, train_data, args)
    
    # Create poisoned data loader
    poisoned_train_loader = DataLoader(
        poisoned_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=functools.partial(collate_fn, max_len=args.seq_len)
    )
    
    # Train victim model on poisoned data
    print("\n=== TRAINING VICTIM MODEL ON POISONED DATA ===")
    optimizer = create_optimizer(
        victim_model.parameters(),
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.optimizer_weight_decay
    )
    CA, ASR = [], []
    
    for epoch in range(args.bd_train_epochs):
        # Train on poisoned data
        train_loss, train_acc = clean_train(
            victim_model, poisoned_train_loader, args, optimizer
        )
        
        # Test attack success rate
        test_loss, clean_test_acc, asr = bd_test(
            victim_model, test_loader, args, trigger_model
        )
        CA.append(clean_test_acc)
        ASR.append(asr)
        
        print(f"[Epoch {epoch+1}/{args.bd_train_epochs}] "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Clean Test Acc={clean_test_acc:.4f}, ASR={asr:.4f}")
    
    final_asr = ASR[-1]
    final_ca = CA[-1]
    
    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print(f"Final Clean Accuracy: {final_ca*100:.2f}%")
    print(f"Final Attack Success Rate (ASR): {final_asr*100:.2f}%")
    print("="*60 + "\n")
    
    model_poison_dic = {
        'CA': CA,
        'ASR': ASR,
    }
    
    # Final comprehensive test epoch with all visualizations
    print("\n" + "="*70)
    print("Starting final comprehensive test epoch...")
    print("="*70)
    
    # Collect sample cases for visualization
    test_loss_final, clean_acc_final, asr_final, sample_cases = bd_test_with_samples(
        model=victim_model,
        loader=test_loader,
        args=args,
        trigger_model=trigger_model,
        max_success=8,
        max_failure=8,
    )
    print(f"Final CA: {clean_acc_final*100:.2f}%, ASR: {asr_final*100:.2f}%")
    
    # Log all results
    exp_dir = log_final_test_epoch(
        model=victim_model,
        trigger_model=trigger_model,
        train_loader=train_loader,
        test_loader=test_loader,
        poisoned_loader=poisoned_train_loader,
        args=args,
        sample_cases=sample_cases,  # Pass pre-collected samples
        poison_indices=poison_indices,  # Pass poisoned sample indices
        trigger_results=None,  # No trigger training results in test mode
        model_poison_dic=model_poison_dic,
        save_dir="Results",
        run_bd_test=False,  # Already ran it above
    )
    print(f"\nAll test results saved to: {exp_dir}")
    
    results = {
        'CA': CA,
        'ASR': ASR,
        'final_ca': final_ca,
        'final_asr': final_asr,
        'exp_dir': exp_dir,
    }
    
    return results
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


def bd_test(model, loader, args, trigger_model=None, mask_model=None, save_plot=False):
    """Test victim model's attack success rate (ASR) and clean accuracy.
    
    Applies triggers to ALL test samples and measures ASR.
    Also computes clean accuracy on original samples.
    
    Args:
        model: Victim model to test
        loader: Test data loader
        args: Test arguments
        trigger_model: Trigger generator (None for basic patch)
        mask_model: Mask generator (for inputaware_masking method)
    
    Returns:
        avg_loss: Average loss
        clean_acc: Clean accuracy
        asr: Attack Success Rate (fraction classified as target)
    """
    model.eval()
    trigger_model.eval() if trigger_model is not None else None
    if mask_model is not None:
        mask_model.eval()
    total_loss = []
    clean_preds = []
    bd_preds = []
    clean_labels = []
    bd_labels = []

    for i, (batch_x, label, padding_mask) in enumerate(loader):
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device).long()
        original_label = label.clone()
        target_labels = (torch.ones_like(label) * args.target_label).long().to(args.device)

        # Apply trigger to all samples
        bd_batch_x, label = apply_trigger(batch_x, label, args, trigger_model, mask_model)

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
    mask_model=None,
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
    if mask_model is not None:
        mask_model.eval()

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

        bd_batch_x, label = apply_trigger(batch_x, label, args, trigger_model, mask_model)

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


if __name__ == "__main__":
    """CLI for testing pre-trained trigger models."""
    from parameters import args_parser
    from utils.tools import load_and_override_params
    
    args = args_parser()
    args = load_and_override_params(args)
    
    args.device = select_least_used_gpu()
    print(f"Using device: {args.device}")
    
    trigger_path = getattr(args, 'trigger_model_path', None)
    
    if trigger_path is None:
        print("Error: Please provide --trigger_model_path argument")
        print("Usage: python test.py --trigger_model_path <path> --root_path <dataset_path>")
        exit(1)
    
    results = test_pretrained_trigger(args, trigger_path)
    print("\nTest completed!")
