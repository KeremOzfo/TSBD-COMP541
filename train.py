"""
Training Module for TimeSeries Backdoor Attack

This module contains all training-related functions:
- Clean training (training_with_clean_data)
- Trigger model training (train_trigger_model)
- Data poisoning (poison_data)
- Model poisoning (poison_model)
- Full backdoor pipeline (training_with_poisoned_data)
"""

import torch
import numpy as np
import functools
import os
from torch.utils.data import DataLoader, Subset

from utils.exp_logging import log_result_clean
from utils.helper_train_test import reconfigure_model_for_data, target_model_dict, trigger_model_dict, create_trigger_model
from utils.helper_data import get_data
from utils.tools import create_optimizer
from src.trigger_training_epochs import trigger_train_epoch, trigger_eval_epoch, train_mask_epoch
from src.methods.inputaware_masking import epoch_inputaware_masking
from src.data_poisoning import poison_with_trigger_all2one, silent_poison_with_trigger_all2one
from test import clean_test, bd_test

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_trigger_training_losses(results, args):
    """Plot all tracked losses and accuracies from trigger training.
    
    Args:
        results: Dictionary containing training results and loss_components
        args: Training arguments (used for output path and method name)
    """
    if not HAS_MATPLOTLIB:
        print("[Warning] matplotlib not available, skipping plots")
        return
    
    epochs = list(range(1, len(results['train_loss']) + 1))
    
    # Create output directory
    output_dir = getattr(args, 'root_path', './outputs')
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot 1: Main metrics (loss, clean acc, ASR)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(epochs, results['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, results['test_loss'], 'r--', label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, results['train_clean_acc'], 'b-', label='Train')
    axes[1].plot(epochs, results['test_clean_acc'], 'r--', label='Test')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Clean Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, results['train_ASR'], 'b-', label='Train')
    axes[2].plot(epochs, results['test_ASR'], 'r--', label='Test')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('ASR')
    axes[2].set_title('Attack Success Rate')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    main_plot_path = os.path.join(plot_dir, f'{args.method}_main_metrics.png')
    plt.savefig(main_plot_path, dpi=150)
    plt.close()
    print(f"  Saved main metrics plot: {main_plot_path}")
    
    # Plot 2: Loss components (if available)
    loss_components = results.get('loss_components', {})
    if loss_components:
        n_components = len(loss_components)
        if n_components > 0:
            n_cols = min(3, n_components)
            n_rows = (n_components + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            
            if n_components == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            
            for idx, (key, values) in enumerate(loss_components.items()):
                if idx < len(axes):
                    axes[idx].plot(epochs, values, 'g-', linewidth=2)
                    axes[idx].set_xlabel('Epoch')
                    axes[idx].set_ylabel('Loss')
                    axes[idx].set_title(key)
                    axes[idx].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(loss_components), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            components_plot_path = os.path.join(plot_dir, f'{args.method}_loss_components.png')
            plt.savefig(components_plot_path, dpi=150)
            plt.close()
            print(f"  Saved loss components plot: {components_plot_path}")


def cal_accuracy(preds, trues):
    """Compute accuracy given predicted labels and true labels."""
    preds_np = np.asarray(preds).reshape(-1)
    trues_np = np.asarray(trues).reshape(-1)
    if preds_np.shape[0] == 0:
        return 0.0
    return float(np.mean(preds_np == trues_np))


def clean_train_epoch(model, loader, args, optimizer):
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
    accuracy = cal_accuracy(predictions, trues.flatten().cpu().numpy())
    
    return avg_loss, accuracy


# Alias for clean_train_epoch
clean_train = clean_train_epoch


def clean_test_epoch(model, loader, args):
    """Test model on clean data for one epoch.
    
    This is a wrapper around test.clean_test for consistency.
    
    Args:
        model: Model to test
        loader: Data loader
        args: Test arguments
    
    Returns:
        avg_loss: Average loss
        accuracy: Clean accuracy
    """
    return clean_test(model, loader, args)


def bd_test_epoch(model, loader, args, trigger_model=None, mask_model=None):
    """Test victim model's attack success rate (ASR) and clean accuracy.
    
    This is a wrapper around test.bd_test for consistency.
    
    Args:
        model: Victim model to test
        loader: Test data loader
        args: Test arguments
        trigger_model: Trigger generator (None for basic patch)
        mask_model: Mask generator (for inputaware_masking method)
    
    Returns:
        avg_loss: Average loss
        clean_acc: Clean accuracy
        asr: Attack Success Rate
    """
    return bd_test(model, loader, args, trigger_model, mask_model)


def warmup_surrogate(model, loader, args):
    """Warmup the surrogate model before trigger training."""
    if args.warmup_epochs == 0:
        return

    print(f"=== WARMUP START ({args.warmup_epochs} epochs) ===")
    optimizer = create_optimizer(
        model.parameters(), 
        optimizer_name=args.surrogate_opt,
        lr=args.surrogate_lr,
        weight_decay=args.surrogate_weight_decay
    )

    for epoch in range(args.warmup_epochs):
        for batch_x, label, padding_mask in loader:

            batch_x = batch_x.float().to(args.device)
            label = label.to(args.device)
            padding_mask = padding_mask.float().to(args.device)

            with torch.no_grad():
                clean_out = model(batch_x, padding_mask, None, None)

            optimizer.zero_grad()

            pred = model(batch_x, padding_mask, None, None)

            loss = torch.nn.functional.mse_loss(pred, clean_out)
            loss.backward()
            optimizer.step()

        print(f"Warmup Epoch {epoch+1}/{args.warmup_epochs}, Loss={loss.item():.4f}")

    print("=== WARMUP DONE ===")


def training_with_clean_data(args):
    """Training a network model with clean data.

    Args:
        args: Training arguments from args_parser()
        
    Returns:
        acc (float): test accuracy
        model (torch.nn): trained model
    """
    print("\n")
    print("******CLEAN TRAINING PIPELINE******")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.root_path}")
    print("\n")
    
    # Load data
    train_data, train_loader = get_data(args, 'train')
    test_data, test_loader = get_data(args, 'test')
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    # Initialize model (sets config params from data and creates fresh model instance)
    model = reconfigure_model_for_data(args, train_data, test_data)
    
    # Optimizer
    optimizer = create_optimizer(
        model.parameters(),
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.optimizer_weight_decay
    )
    
    best_acc = 0.0
    # Training history for logging
    train_history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }
    
    print(f"\nStarting training for {args.train_epochs} epochs:\n")
    
    for epoch in range(args.train_epochs):
        # Train
        train_loss, train_acc = clean_train_epoch(model, train_loader, args, optimizer)
        
        # Test
        test_loss, test_acc = clean_test_epoch(model, test_loader, args)
        
        # Record history
        train_history["train_loss"].append(train_loss)
        train_history["test_loss"].append(test_loss)
        train_history["train_acc"].append(train_acc)
        train_history["test_acc"].append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        print(f"[Epoch {epoch+1}/{args.train_epochs}] "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
    
    final_acc = train_history["test_acc"][-1]
    
    print("\n")
    print("CLEAN TRAINING COMPLETED")
    print(f"Final Test Accuracy: {final_acc*100:.2f}%")
    print(f"Best Test Accuracy: {best_acc*100:.2f}%")
    print("\n")
    
    # Log results with training curves
    log_result_clean(args, final_acc, train_history=train_history)
    
    return final_acc, model


def train_trigger_model(trigger_model, surrogate_model, train_loader, args, train_data=None):
    """Train the dynamic trigger generator model.
    
    Uses the standard backdoor training approach with a surrogate classifier:
    - Trains trigger generator to create triggers that fool the surrogate
    - Trains surrogate to maintain clean accuracy while being vulnerable to triggers
    
    Different methods have different requirements:
    - basic, marksman, frequency: single loader, no extra models
    - diversity: optional second loader for diversity loss
    - ultimate, inputaware: second loader + previous trigger model for stable training
    
    Args:
        trigger_model (torch.nn): the trigger generator network
        surrogate_model (torch.nn): the surrogate classification model
        train_loader: DataLoader for training data
        args: Training arguments (includes args.method for method selection)
        train_data: Optional training dataset (for creating second loader)
    
    Returns:
        dict: Training results with losses and accuracies
    """
    print(f"=== STEP 1: TRIGGER MODEL TRAINING (method={args.method}) ===")

    trigger_batch_size = getattr(args, 'trigger_batch_size', None)
    if trigger_batch_size is None or trigger_batch_size <= 0:
        trigger_batch_size = args.batch_size

    if train_loader is not None and getattr(train_loader, 'batch_size', None) != trigger_batch_size:
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=trigger_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
            collate_fn=train_loader.collate_fn
        )
    
    optimizer_trigger = create_optimizer(
        trigger_model.parameters(),
        optimizer_name=args.trigger_opt,
        lr=args.trigger_lr,
        weight_decay=args.trigger_weight_decay
    )
    optimizer_surrogate = create_optimizer(
        surrogate_model.parameters(),
        optimizer_name=args.surrogate_opt,
        lr=args.surrogate_lr,
        weight_decay=args.surrogate_weight_decay
    )
    
    trigger_epochs = args.trigger_epochs
    patience = getattr(args, 'trigger_patience', 0)
    train_losses, train_clean_accs, train_ASR = [], [], []
    test_losses, test_clean_accs, test_ASR = [], [], []
    all_loss_dicts = []  # Accumulate loss_dict from each epoch
    
    # Early stopping variables
    best_test_asr = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    
    # For methods that need a copy of trigger model (ultimate, inputaware, pureinputaware)
    trigger_model_prev = None
    methods_needing_prev = ['ultimate', 'inputaware', 'pureinputaware']
    if args.method in methods_needing_prev:
        import copy
        trigger_model_prev = copy.deepcopy(trigger_model)
        trigger_model_prev.eval()
        print(f"  Created trigger_model_prev for {args.method} method")
    
    # For methods that need a second loader (diversity, ultimate, inputaware, pureinputaware)
    loader2 = None
    methods_needing_loader2 = ['diversity', 'ultimate', 'inputaware', 'pureinputaware']
    if args.method in methods_needing_loader2 and train_data is not None:
        from data_provider.uea import collate_fn
        loader2 = DataLoader(
            train_data,
            batch_size=trigger_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False,
            collate_fn=functools.partial(collate_fn, max_len=args.seq_len)
        )
        print(f"  Created secondary loader for {args.method} method (diversity/cross-trigger)")
    
    # Warmup surrogate model before trigger training
    warmup_surrogate(surrogate_model, train_loader, args)
    
    for epoch in range(trigger_epochs):
        train_loss, loss_dict, clean_acc, bd_acc = trigger_train_epoch(
            trigger_model=trigger_model,
            surrogate_model=surrogate_model,
            loader=train_loader,
            args=args,
            optimizer_trigger=optimizer_trigger,
            optimizer_surrogate=optimizer_surrogate,
            trigger_model_prev=trigger_model_prev,
            loader2=loader2
        )
        test_loss, test_clean_acc, test_bd_acc = bd_test_epoch(
            surrogate_model, train_loader, args, trigger_model
        )
        
        print(f"[Trigger Epoch {epoch+1}/{trigger_epochs}] "
              f"Loss={train_loss:.4f}, Clean Acc={clean_acc:.4f}, BD Acc (ASR)={bd_acc:.4f}")
        print(f"    [Test] Loss={test_loss:.4f}, Clean Acc={test_clean_acc:.4f}, BD Acc (ASR)={test_bd_acc:.4f}")
        
        train_losses.append(train_loss)
        train_clean_accs.append(clean_acc)
        train_ASR.append(bd_acc)
        test_losses.append(test_loss)
        test_clean_accs.append(test_clean_acc)
        test_ASR.append(test_bd_acc)
        all_loss_dicts.append(loss_dict)
        
        # Early stopping logic
        if patience > 0:
            if test_bd_acc > best_test_asr:
                best_test_asr = test_bd_acc
                best_epoch = epoch
                epochs_without_improvement = 0
                print(f"    [Early Stopping] New best ASR: {best_test_asr:.4f}")
            else:
                epochs_without_improvement += 1
                print(f"    [Early Stopping] No improvement for {epochs_without_improvement}/{patience} epochs")
                
                if epochs_without_improvement >= patience:
                    print(f"\n=== EARLY STOPPING TRIGGERED ===")
                    print(f"No improvement in test ASR for {patience} epochs")
                    print(f"Best ASR: {best_test_asr:.4f} at epoch {best_epoch+1}")
                    print(f"Stopping at epoch {epoch+1}/{trigger_epochs}")
                    break
    
    # Aggregate loss_dict values across epochs (compute mean per epoch)
    aggregated_losses = {}
    if all_loss_dicts:
        # Get all keys from first loss_dict
        all_keys = set()
        for ld in all_loss_dicts:
            all_keys.update(ld.keys())
        
        for key in all_keys:
            aggregated_losses[key] = []
            for ld in all_loss_dicts:
                if key in ld and ld[key]:
                    aggregated_losses[key].append(np.mean(ld[key]))
                else:
                    aggregated_losses[key].append(0.0)
    
    results = {
        'train_loss': train_losses,
        'train_clean_acc': train_clean_accs,
        'train_ASR': train_ASR,
        'test_loss': test_losses,
        'test_clean_acc': test_clean_accs,
        'test_ASR': test_ASR,
        'loss_components': aggregated_losses,
        'trigger_model': trigger_model  # Include model for saving
    }
    print("=== TRIGGER MODEL TRAINING COMPLETED ===")
    print(f"Final Trigger ASR: {test_bd_acc:.4f}")
    
    # Plot loss curves
    plot_trigger_training_losses(results, args)
    
    return results


def train_trigger_model_with_mask(trigger_model, surrogate_model, train_loader, args, train_data=None):
    """
    Train trigger model with mask network (full Input-Aware implementation).
    
    Two-stage training:
    1. Pre-train mask network with diversity + sparsity
    2. Train pattern generator + classifier with frozen mask
    
    Args:
        trigger_model: Pattern generator
        surrogate_model: Classifier
        train_loader: Training data loader
        args: Configuration
        train_data: Dataset for creating secondary loader
    
    Returns:
        Dictionary with training results and mask model
    """
    print("=== STEP 1: TRIGGER MODEL TRAINING (method=inputaware_masking) ===")
    print("  Stage 1: Pre-training mask network...")
    
    # Create mask network
    from src.models.trigger_models.mask_generator import create_mask_generator
    mask_model = create_mask_generator(args, use_transformer=False).to(args.device)
    
    # Mask pre-training settings
    mask_pretrain_epochs = getattr(args, 'mask_pretrain_epochs', 25)
    
    # Create optimizer for mask
    optimizer_mask = create_optimizer(
        mask_model.parameters(),
        optimizer_name=getattr(args, 'mask_opt', 'adam'),
        lr=getattr(args, 'mask_lr', 0.001),
        weight_decay=getattr(args, 'mask_weight_decay', 0.0)
    )
    
    # Create secondary loader for diversity
    from data_provider.uea import collate_fn
    loader2 = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=functools.partial(collate_fn, max_len=args.seq_len)
    )
    
    # Pre-train mask network
    for epoch in range(mask_pretrain_epochs):
        train_loss, loss_dict = train_mask_epoch(
            mask_model=mask_model,
            loader1=train_loader,
            loader2=loader2,
            args=args,
            opt_mask=optimizer_mask,
            train=True
        )
        
        # Get average sparsity
        with torch.no_grad():
            sample_batch = next(iter(train_loader))
            sample_x = sample_batch[0].float().to(args.device)
            sample_mask = mask_model(sample_x)
            avg_sparsity = mask_model.get_sparsity(sample_mask).item()
        
        print(f"  [Mask Epoch {epoch+1}/{mask_pretrain_epochs}] "
              f"Loss={train_loss:.4f}, Sparsity={avg_sparsity:.4f}, "
              f"Div={np.mean(loss_dict['L_div']):.4f}, Norm={np.mean(loss_dict['L_norm']):.4f}")
    
    # Freeze mask network
    mask_model.eval()
    for param in mask_model.parameters():
        param.requires_grad = False
    print(f"  Mask network pre-trained and frozen (final sparsity: {avg_sparsity:.4f})")
    
    # Stage 2: Train pattern generator and classifier
    print("  Stage 2: Training pattern generator and classifier...")
    
    # Create optimizers
    optimizer_trigger = create_optimizer(
        trigger_model.parameters(),
        optimizer_name=args.trigger_opt,
        lr=args.trigger_lr,
        weight_decay=args.trigger_weight_decay
    )
    optimizer_surrogate = create_optimizer(
        surrogate_model.parameters(),
        optimizer_name=args.surrogate_opt,
        lr=args.surrogate_lr,
        weight_decay=args.surrogate_weight_decay
    )
    
    trigger_epochs = args.trigger_epochs
    patience = getattr(args, 'trigger_patience', 0)
    train_losses, train_clean_accs, train_ASR = [], [], []
    test_losses, test_clean_accs, test_ASR = [], [], []
    all_loss_dicts = []
    
    # Early stopping variables
    best_test_asr = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    
    for epoch in range(trigger_epochs):
        train_loss, loss_dict, clean_acc, bd_acc = epoch_inputaware_masking(
            bd_model=trigger_model,
            mask_model=mask_model,
            surr_model=surrogate_model,
            loader1=train_loader,
            args=args,
            loader2=loader2,
            opt_trig=optimizer_trigger,
            opt_class=optimizer_surrogate,
            train=True,
            mask_frozen=True
        )
        
        test_loss, test_clean_acc, test_bd_acc = bd_test_epoch(
            surrogate_model, train_loader, args, trigger_model, mask_model=mask_model
        )
        
        print(f"[Trigger Epoch {epoch+1}/{trigger_epochs}] "
              f"Loss={train_loss:.4f}, Clean Acc={clean_acc:.4f}, BD Acc (ASR)={bd_acc:.4f}")
        print(f"    [Test] Loss={test_loss:.4f}, Clean Acc={test_clean_acc:.4f}, BD Acc (ASR)={test_bd_acc:.4f}")
        
        train_losses.append(train_loss)
        train_clean_accs.append(clean_acc)
        train_ASR.append(bd_acc)
        test_losses.append(test_loss)
        test_clean_accs.append(test_clean_acc)
        test_ASR.append(test_bd_acc)
        all_loss_dicts.append(loss_dict)
        
        # Early stopping logic
        if patience > 0:
            if test_bd_acc > best_test_asr:
                best_test_asr = test_bd_acc
                best_epoch = epoch
                epochs_without_improvement = 0
                print(f"    [Early Stopping] New best ASR: {best_test_asr:.4f}")
            else:
                epochs_without_improvement += 1
                print(f"    [Early Stopping] No improvement for {epochs_without_improvement}/{patience} epochs")
                
                if epochs_without_improvement >= patience:
                    print(f"\n=== EARLY STOPPING TRIGGERED ===")
                    print(f"No improvement in test ASR for {patience} epochs")
                    print(f"Best ASR: {best_test_asr:.4f} at epoch {best_epoch+1}")
                    print(f"Stopping at epoch {epoch+1}/{trigger_epochs}")
                    break
    
    # Aggregate losses
    aggregated_losses = {}
    if all_loss_dicts:
        all_keys = set()
        for ld in all_loss_dicts:
            all_keys.update(ld.keys())
        
        for key in all_keys:
            aggregated_losses[key] = []
            for ld in all_loss_dicts:
                if key in ld and ld[key]:
                    aggregated_losses[key].append(np.mean(ld[key]))
                else:
                    aggregated_losses[key].append(0.0)
    
    results = {
        'train_loss': train_losses,
        'train_clean_acc': train_clean_accs,
        'train_ASR': train_ASR,
        'test_loss': test_losses,
        'test_clean_acc': test_clean_accs,
        'test_ASR': test_ASR,
        'loss_components': aggregated_losses,
        'mask_model': mask_model  # Return mask model for poisoning
    }
    print("=== TRIGGER MODEL TRAINING COMPLETED ===")
    print(f"Final Trigger ASR: {test_bd_acc:.4f}")
    
    # Plot loss curves
    plot_trigger_training_losses(results, args)
    
    return results


def poison_data(trigger_model, train_data, args, mask_model=None):
    """Apply triggers to training data to create poisoned samples.
    
    Delegates to either normal or silent poisoning based on args.use_silent_poisoning.
    
    Args:
        trigger_model (torch.nn): trained trigger generator
        train_data: Original training dataset
        args: Arguments containing poisoning configuration:
            - use_silent_poisoning: if True, use silent poisoning (clean-label backdoor)
            - poisoning_ratio: fraction of samples to poison
            - target_label: target class for backdoor
            - lambda_ratio: fraction keeping original labels (for silent mode)
        mask_model: Optional mask model (for inputaware_masking method)
    
    Returns:
        poisoned_dataset: Dataset with poisoned samples
        poison_indices: List of poisoned sample indices
        (For silent mode, also returns clean_label_indices and target_label_indices)
    """
    print("=== STEP 2: DATA POISONING ===")
    
    # Check if using silent poisoning
    use_silent = getattr(args, 'use_silent_poisoning', False)
    
    if use_silent:
        # Use silent poisoning (clean-label backdoor)
        print("Using SILENT POISONING mode (clean-label backdoor)")
        train_data, poison_indices, clean_label_indices, target_label_indices = \
            silent_poison_with_trigger_all2one(trigger_model, train_data, args)
        
        # Store additional indices for logging
        args._clean_label_indices = clean_label_indices
        args._target_label_indices = target_label_indices
        
        return train_data, poison_indices
    else:
        # Use normal poisoning (all samples relabeled to target)
        print("Using NORMAL POISONING mode (all samples -> target label)")
        train_data, poison_indices = \
            poison_with_trigger_all2one(trigger_model, train_data, args)
        
        return train_data, poison_indices


def poison_model(model, trigger_model, train_loader, test_loader, args):
    """Train the main model with poisoned data (backdoor injection).
    
    Since data is already poisoned in Step 2, we use clean training here.
    For evaluation, we still use bd_test to measure attack success rate.
    
    Args:
        model (torch.nn): the target model for backdoor insertion
        trigger_model (torch.nn): trained network model for trigger generation
        train_loader: Training data loader (contains poisoned samples)
        test_loader: Test data loader
        args: Training arguments
    
    Returns:
        dict: Results with CA and ASR over epochs
    """
    print("=== STEP 3: MODEL POISONING ===")
    print("Training model on poisoned dataset...")
    
    optimizer = create_optimizer(
        model.parameters(),
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.optimizer_weight_decay
    )
    CA, ASR = [], []
    model_poison_dic = {}
    
    for epoch in range(args.bd_train_epochs):
        # Train on the poisoned dataset (using clean_train since data is pre-poisoned)
        train_loss, train_acc = clean_train(
            model, train_loader, args, optimizer
        )
        
        # Test attack success rate and clean accuracy using bd_test
        test_loss, clean_test_acc, asr = bd_test(
            model, test_loader, args, trigger_model
        )
        CA.append(clean_test_acc)
        ASR.append(asr)
        
        print(f"[Poison Epoch {epoch+1}/{args.bd_train_epochs}] "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Clean Test Acc={clean_test_acc:.4f}, ASR={asr:.4f}")
    
    final_asr = ASR[-1]
    print("=== MODEL POISONING COMPLETED ===")
    print(f"Final Attack Success Rate (ASR): {final_asr:.4f}")
    model_poison_dic['CA'] = CA
    model_poison_dic['ASR'] = ASR
    return model_poison_dic


def train_defeat(trigger_model, surrogate_model, train_loader, args, train_data=None):
    """
    Train trigger model using DEFEAT method with bilevel optimization.
    
    DEFEAT (Deep Hidden Feature Backdoor Attacks) uses:
    1. Clean model pre-training
    2. Auxiliary logit training for feature hiding
    3. Alternating optimization between trigger and classifier
    
    Args:
        trigger_model: Trigger generator T_φ
        surrogate_model: Classifier F_θ  
        train_loader: Training data loader
        args: Configuration
        train_data: Training dataset (optional)
    
    Returns:
        dict: Training results
    """
    from src.methods.defeat import train_defeat as defeat_training
    
    print("="*70)
    print("=== TRIGGER MODEL TRAINING (method=defeat) ===")
    print("="*70)
    print(f"DEFEAT Configuration:")
    print(f"  - β₁ (clean weight): {getattr(args, 'beta_1', 1.0)}")
    print(f"  - β₂ (adversarial weight): {getattr(args, 'beta_2', 0.1)}")
    print(f"  - ε (stealthiness budget): {getattr(args, 'epsilon_budget', 0.1)}")
    print(f"  - R (alternating iterations): {getattr(args, 'defeat_iterations', 20)}")
    print(f"  - Clean pretrain epochs: {getattr(args, 'clean_pretrain_epochs', 50)}")
    print(f"  - Aux logit epochs: {getattr(args, 'aux_logit_epochs', 10)}")
    print()
    
    # Run DEFEAT training
    results = defeat_training(trigger_model, surrogate_model, train_loader, args, train_data)
    
    print(f"\nFinal Trigger ASR: {results['final_bd_acc']:.4f}")
    print(f"Final Clean Accuracy: {results['final_clean_acc']:.4f}")
    
    # Store aux_logits for later use
    results['method'] = 'defeat'
    
    return results
