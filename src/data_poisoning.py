"""
Data Poisoning Module for TimeSeries Backdoor Attack

This module contains data poisoning functions:
- poison_with_trigger_all2one: Poison dataset with a single target label
- poison_with_trigger_one2one: Poison dataset with mapped target labels per class
- silent_poison_with_trigger_all2one: Poison with hard/soft examples to reduce visibility
"""

import torch
import random
import numpy as np


def poison_with_trigger_all2one(trigger_model, train_data, args):
    """Given the trigger network, poison the dataset with a single target label.
    
    All poisoned samples get relabeled to the same target label (args.target_label).
    
    Args:
        trigger_model (torch.nn): trained trigger generator network
        train_data: Original training dataset
        args: Arguments containing:
            - poisoning_ratio: fraction of samples to poison
            - target_label: the target class for all poisoned samples
            - device: torch device
            - clip_ratio: trigger magnitude for basic patch (if no trigger_model)
    
    Returns:
        train_data: Dataset with poisoned samples (modified in-place)
        poison_indices: List of poisoned sample indices
    """
    print("=== ALL2ONE DATA POISONING ===")
    print(f"Poisoning ratio: {args.poisoning_ratio*100:.1f}%")
    print(f"Target label: {args.target_label}")
    
    # Calculate number of samples to poison
    total_samples = len(train_data)
    num_poison = int(total_samples * args.poisoning_ratio)
    
    if num_poison == 0:
        print("No samples to poison (ratio too low)")
        print("=== DATA POISONING SKIPPED ===")
        return train_data, []
    
    # Filter out samples that already have the target label
    # Only poison samples with non-target labels
    non_target_indices = []
    for idx in range(total_samples):
        sample_data = train_data[idx]
        # Extract label
        if isinstance(sample_data, tuple):
            y = sample_data[1]
        else:
            y = train_data.labels_df.loc[train_data.all_IDs[idx]]
        
        # Convert to int if needed
        if isinstance(y, torch.Tensor):
            y = y.item()
        elif isinstance(y, np.ndarray):
            y = int(y)
        
        # Only include non-target samples
        if y != args.target_label:
            non_target_indices.append(idx)
    
    print(f"Total samples: {total_samples}")
    print(f"Non-target samples available: {len(non_target_indices)}")
    
    # Adjust num_poison if not enough non-target samples
    if num_poison > len(non_target_indices):
        print(f"WARNING: Requested {num_poison} poisoned samples, but only {len(non_target_indices)} non-target samples available")
        num_poison = len(non_target_indices)
    
    # Sample from non-target indices only
    poison_indices = random.sample(non_target_indices, num_poison)
    
    print(f"Poisoning {num_poison} out of {len(non_target_indices)} non-target samples")
    
    # Make DataFrames writable
    train_data.feature_df = train_data.feature_df.copy()
    train_data.labels_df = train_data.labels_df.copy()
    
    # Apply triggers to selected samples
    if trigger_model is not None:
        trigger_model.eval()
        print("Using dynamic trigger model to poison data...")
    else:
        print("Using basic patch trigger to poison data...")
    
    # Poison the dataset in-place by modifying data and labels
    with torch.no_grad():
        for idx in poison_indices:
            # Get the sample
            sample_data = train_data[idx]
            
            # Extract components based on dataset structure
            if isinstance(sample_data, tuple):
                x, y = sample_data[0], sample_data[1]
            else:
                x = sample_data
                y = None
            
            # Convert to tensor if needed
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            
            x = x.unsqueeze(0).to(args.device)  # Add batch dimension
            x = x.float()  # Ensure float32
            
            # Store original length for variable-length sequences
            original_len = x.shape[1]
            
            # Apply trigger based on mode
            if trigger_model is not None:
                # Dynamic trigger using the trained trigger model
                target_label = torch.tensor([args.target_label]).to(args.device)
                
                # Pad input to seq_len if needed (for variable-length datasets)
                if original_len < args.seq_len:
                    x_padded = torch.zeros(1, args.seq_len, x.shape[2], device=args.device)
                    x_padded[:, :original_len, :] = x
                    # Create padding mask: 1 for valid positions, 0 for padded positions
                    padding_mask = torch.zeros(1, args.seq_len, device=args.device)
                    padding_mask[:, :original_len] = 1.0
                else:
                    x_padded = x[:, :args.seq_len, :]
                    # No padding needed, all positions are valid
                    padding_mask = torch.ones(1, args.seq_len, device=args.device)
                
                # Generate trigger with padding mask to avoid triggers on padded positions
                _, trigger_clip = trigger_model(x_padded, padding_mask, None, None, target_label)
                
                # Extract trigger for original length only
                trigger_clip = trigger_clip[:, :original_len, :]
                
                # Apply trigger to original (unpadded) data
                x_poisoned = x + trigger_clip
            else:
                # Basic patch trigger
                x_poisoned = x.clone()
                x_poisoned[0, -5:, 0] += args.clip_ratio  # Apply basic patch
            
            # Update the dataset with poisoned sample
            x_poisoned = x_poisoned.squeeze(0).cpu()
            
            # Modify the dataset entry
            train_data.feature_df.loc[train_data.all_IDs[idx]] = x_poisoned.numpy()
            
            # Update label to target label (all2one - same target for all)
            train_data.labels_df.loc[train_data.all_IDs[idx]] = args.target_label
    
    print(f"Successfully poisoned {num_poison} samples -> target label {args.target_label}")
    print("=== ALL2ONE DATA POISONING COMPLETED ===")
    
    return train_data, poison_indices



def silent_poison_with_trigger_all2one(trigger_model, train_data, args):
    """Given the trigger network, poison the dataset with reduced visibility using clean-label backdoors.
    
    Uses a lambda parameter to control what fraction of poisoned samples keep their original label.
    This makes the attack harder to detect via label inspection while maintaining effectiveness.
    
    - All selected samples receive the trigger pattern
    - (1 - lambda) fraction get relabeled to target_label (standard backdoor)
    - lambda fraction keep their original label (clean-label backdoor)
    
    Args:
        trigger_model (torch.nn): trained trigger generator network
        train_data: Original training dataset
        args: Arguments containing:
            - poisoning_ratio: fraction of samples to poison
            - target_label: the target class for poisoned samples
            - device: torch device
            - clip_ratio: trigger magnitude for basic patch (if no trigger_model)
            - lambda_ratio: fraction of backdoor samples that keep original label (default: 0.2)
    
    Returns:
        train_data: Dataset with poisoned samples (modified in-place)
        poison_indices: List of all poisoned sample indices
        clean_label_indices: List of indices that kept original labels
        target_label_indices: List of indices relabeled to target
    """
    print("=== SILENT ALL2ONE DATA POISONING ===")
    print(f"Poisoning ratio: {args.poisoning_ratio*100:.1f}%")
    print(f"Target label: {args.target_label}")
    
    # Get lambda parameter - fraction of poisoned samples keeping original label
    lambda_ratio = getattr(args, 'lambda_ratio', 0.2)
    
    print(f"Lambda ratio: {lambda_ratio*100:.1f}% (clean-label backdoors)")
    
    # Calculate number of samples to poison
    total_samples = len(train_data)
    num_poison = int(total_samples * args.poisoning_ratio)
    
    if num_poison == 0:
        print("No samples to poison (ratio too low)")
        print("=== DATA POISONING SKIPPED ===")
        return train_data, [], [], []
    
    # Filter out samples that already have the target label
    # Only poison samples with non-target labels
    non_target_indices = []
    for idx in range(total_samples):
        sample_data = train_data[idx]
        # Extract label
        if isinstance(sample_data, tuple):
            y = sample_data[1]
        else:
            y = train_data.labels_df.loc[train_data.all_IDs[idx]]
        
        # Convert to int if needed
        if isinstance(y, torch.Tensor):
            y = y.item()
        elif isinstance(y, np.ndarray):
            y = int(y)
        
        # Only include non-target samples
        if y != args.target_label:
            non_target_indices.append(idx)
    
    print(f"Total samples: {total_samples}")
    print(f"Non-target samples available: {len(non_target_indices)}")
    
    # Adjust num_poison if not enough non-target samples
    if num_poison > len(non_target_indices):
        print(f"WARNING: Requested {num_poison} poisoned samples, but only {len(non_target_indices)} non-target samples available")
        num_poison = len(non_target_indices)
    
    # Sample from non-target indices only
    poison_indices = random.sample(non_target_indices, num_poison)
    
    # Split into clean-label and target-label samples based on lambda
    num_clean_label = int(num_poison * lambda_ratio)
    num_target_label = num_poison - num_clean_label
    
    random.shuffle(poison_indices)
    clean_label_indices = poison_indices[:num_clean_label]
    target_label_indices = poison_indices[num_clean_label:]
    
    print(f"Poisoning {num_poison} non-target samples: {num_target_label} target-label, {num_clean_label} clean-label")
    
    # Make DataFrames writable
    train_data.feature_df = train_data.feature_df.copy()
    train_data.labels_df = train_data.labels_df.copy()
    
    # Apply triggers to selected samples
    if trigger_model is not None:
        trigger_model.eval()
        print("Using dynamic trigger model to poison data...")
    else:
        print("Using basic patch trigger to poison data...")
    
    # Poison the dataset in-place by modifying data and labels
    with torch.no_grad():
        for idx in poison_indices:
            is_clean_label = idx in clean_label_indices
            
            # Get the sample
            sample_data = train_data[idx]
            
            # Extract components based on dataset structure
            if isinstance(sample_data, tuple):
                x, y = sample_data[0], sample_data[1]
            else:
                x = sample_data
                y = None
            
            # Store original label before any modifications
            original_label = int(train_data.labels_df.loc[train_data.all_IDs[idx]])
            
            # Convert to tensor if needed
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            
            x = x.unsqueeze(0).to(args.device)  # Add batch dimension
            x = x.float()  # Ensure float32
            
            # Store original length for variable-length sequences
            original_len = x.shape[1]
            
            # Apply trigger based on mode (all samples get trigger)
            if trigger_model is not None:
                # Dynamic trigger using the trained trigger model
                target_label = torch.tensor([args.target_label]).to(args.device)
                
                # Pad input to seq_len if needed (for variable-length datasets)
                if original_len < args.seq_len:
                    x_padded = torch.zeros(1, args.seq_len, x.shape[2], device=args.device)
                    x_padded[:, :original_len, :] = x
                    # Create padding mask: 1 for valid positions, 0 for padded positions
                    padding_mask = torch.zeros(1, args.seq_len, device=args.device)
                    padding_mask[:, :original_len] = 1.0
                else:
                    x_padded = x[:, :args.seq_len, :]
                    # No padding needed, all positions are valid
                    padding_mask = torch.ones(1, args.seq_len, device=args.device)
                
                # Generate trigger with padding mask to avoid triggers on padded positions
                _, trigger_clip = trigger_model(x_padded, padding_mask, None, None, target_label)
                
                # Extract trigger for original length only
                trigger_clip = trigger_clip[:, :original_len, :]
                
                # Apply trigger to original (unpadded) data
                x_poisoned = x + trigger_clip
            else:
                # Basic patch trigger
                x_poisoned = x.clone()
                x_poisoned[0, -5:, 0] += args.clip_ratio
            
            # Update the dataset with poisoned sample
            x_poisoned = x_poisoned.squeeze(0).cpu()
            
            # Modify the dataset entry (trigger always applied)
            train_data.feature_df.loc[train_data.all_IDs[idx]] = x_poisoned.numpy()
            
            # Update label based on clean-label or target-label
            if is_clean_label:
                # Keep original label (clean-label backdoor)
                train_data.labels_df.loc[train_data.all_IDs[idx]] = original_label
            else:
                # Change to target label (standard backdoor)
                train_data.labels_df.loc[train_data.all_IDs[idx]] = args.target_label
    
    print(f"Successfully poisoned {num_poison} samples")
    print(f"  Target-label samples: {num_target_label} -> label {args.target_label}")
    print(f"  Clean-label samples: {num_clean_label} (original labels preserved)")
    print("=== SILENT ALL2ONE DATA POISONING COMPLETED ===")
    
    return train_data, poison_indices, clean_label_indices, target_label_indices


