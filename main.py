import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import functools
from utils.logging import log_result_clean, log_all
from data_provider.data_factory import data_provider
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from models.TimesNet import Model as TimesNet
from models.LSTM import LSTMClassifier as LSTM
from models.PatchTST_Classifier import PatchTST_Classifier
from models.iTransformer import Model as iTransformer
from models.TimeMixer import Model as TimeMixer
from models.bd_TimesNet import Model as Bd_TimesNet
from models.bd_patchTST import Model as Bd_PatchTST
from models.bd_inverted import Model as Bd_Inverted
from models.bd_cnn import bd_CNN
from epoch import (
    clean_train, clean_test, bd_test,
    clean_train_epoch, clean_test_epoch, bd_test_epoch, bd_test_with_samples
)
from dynamic_triggers import (trigger_train_epoch
)

#Notes#
# backdoor training consist of 3 phases not sure these are avilable here!!!!!!
# Functions do not have any clear definition what is subsetdataset ????
# why for warmup we use trigger model???
# models
import time
# args
from parameters import args_parser

target_model_dict = {
    'timesnet': TimesNet,
    'lstm': LSTM,
    'patchtst': PatchTST_Classifier,
    'itransformer': iTransformer,
    'timemixer': TimeMixer
}

trigger_model_dict = {
    'timesnet': Bd_TimesNet,
    'cnn': bd_CNN,
    'patchtst': Bd_PatchTST,
    'itst': Bd_Inverted
}

def trigger_train_epoch_global():
    return


def select_least_used_gpu():
    """Select the CUDA device with the least used memory.
    
    Returns:
        torch.device: The device with least used memory, or CPU if no CUDA available.
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    num_devices = torch.cuda.device_count()
    if num_devices == 1:
        return torch.device('cuda:0')
    
    # Get memory usage for each device
    device_memory_usage = []
    for i in range(num_devices):
        try:
            free, total = torch.cuda.mem_get_info(i)
            used = total - free
            device_memory_usage.append((i, used))
        except Exception as e:
            # If we can't get memory info, assume device is available with 0 usage
            device_memory_usage.append((i, 0))
    
    # Sort by memory usage (ascending) and return the device with least usage
    device_memory_usage.sort(key=lambda x: x[1])
    least_used_device_id = device_memory_usage[0][0]
    
    return torch.device(f'cuda:{least_used_device_id}')


def subset_dataset(dataset, ratio):
    """Create a subset of the dataset based on ratio.
    
    Args:
        dataset: Original dataset
        ratio: Fraction of data to keep (0.0 to 1.0)
    
    Returns:
        Subset of the dataset
    """
    if ratio >= 1.0:
        return dataset
    import random
    idx = list(range(len(dataset)))
    random.shuffle(idx)
    cut = int(len(idx) * ratio)
    selected = idx[:cut]
    return torch.utils.data.Subset(dataset, selected)


def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader

def get_clean_model(args, train_data, test_data,model_type=None):
    # seq_len (padding mask için)
    args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)

    # classification: pred_len = 0
    args.pred_len = 0

    # feature dimension
    args.enc_in = train_data.feature_df.shape[1]

    # NUMBER OF CLASSES (CORRECT)
    args.num_class = int(train_data.labels_df.nunique().values[0])

    # Model init
    if model_type is not None:
        model = target_model_dict[model_type.lower()](args).float().to(args.device)
    else:
        model = target_model_dict[args.model.lower()](args).float().to(args.device)

    return model


def warmup_surrogate(model, loader, args):
    if args.warmup_epochs == 0:
        return

    print(f"=== WARMUP START ({args.warmup_epochs} epochs) ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.surrogate_lr)

    for epoch in range(args.warmup_epochs):
        for batch_x, label, padding_mask in loader:

            batch_x = batch_x.float().to(args.device)
            label = label.to(args.device)
            padding_mask = padding_mask.float().to(args.device)

            with torch.no_grad():
                clean_out = model(batch_x, padding_mask, None, None)

            optimizer.zero_grad()

            pred, _ = model(batch_x, padding_mask, None, None, label)

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
        acc: Final test accuracy
        model: Trained model
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
    model = get_clean_model(args, train_data, test_data)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
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


def train_trigger_model(trigger_model, surrogate_model, train_loader, args):
    """Train the dynamic trigger generator model.
    
    Uses the standard backdoor training approach with a surrogate classifier:
    - Trains trigger generator to create triggers that fool the surrogate
    - Trains surrogate to maintain clean accuracy while being vulnerable to triggers
    
    Args:
        trigger_model: The trigger generator network (e.g., Bd_MLP, TimesBA)
        surrogate_model: The surrogate classification model
        train_loader: DataLoader for training data
        args: Training arguments
    
    Returns:
        Trained trigger model
    """
    print("=== STEP 1: TRIGGER MODEL TRAINING ===")
    
    # Initialize optimizers for both trigger generator and surrogate
    optimizer_trigger = torch.optim.Adam(trigger_model.parameters(), lr=args.surrogate_lr)
    optimizer_surrogate = torch.optim.Adam(surrogate_model.parameters(), lr=args.surrogate_lr)
    
    trigger_epochs = getattr(args, 'trigger_epochs', 10)
    train_losses, train_clean_accs, train_ASR = [], [], []
    test_losses, test_clean_accs, test_ASR = [], [], []
    
    # Train trigger model using the proper trigger training epoch function
    for epoch in range(trigger_epochs):
        train_loss, clean_acc, bd_acc = trigger_train_epoch(
            trigger_model=trigger_model,
            surrogate_model=surrogate_model,
            loader=train_loader,
            args=args,
            optimizer_trigger=optimizer_trigger,
            optimizer_surrogate=optimizer_surrogate
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
    
    results = {
        'train_loss': train_losses,
        'train_clean_acc': train_clean_accs,
        'train_ASR': train_ASR,
        'test_loss': test_losses,
        'test_clean_acc': test_clean_accs,
        'test_ASR': test_ASR
    }
    print("=== TRIGGER MODEL TRAINING COMPLETED ===")
    print(f"Final Trigger ASR: {test_bd_acc:.4f}")
    
    return results
    


def poison_data(trigger_model, train_data, args):
    """Apply triggers to training data to create poisoned samples.
    
    Args:
        trigger_model: Trained trigger generator (None for basic mode)
        train_data: Original training dataset
        args: Arguments containing poisoning ratio and target label
    
    Returns:
        poisoned_dataset, poison_indices
    """
    print("=== STEP 2: DATA POISONING ===")
    print(f"Poisoning ratio: {args.poisoning_ratio*100:.1f}%")
    print(f"Target label: {args.target_label}")
    
    # Calculate number of samples to poison
    total_samples = len(train_data)
    num_poison = int(total_samples * args.poisoning_ratio)
    
    if num_poison == 0:
        print("No samples to poison (ratio too low)")
        print("=== DATA POISONING SKIPPED ===")
        return train_data, []
    
    # Select random indices to poison
    import random
    poison_indices = random.sample(range(total_samples), num_poison)
    poison_indices_set = set(poison_indices)
    
    print(f"Poisoning {num_poison} out of {total_samples} samples")
    
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
            
            # Apply trigger based on mode
            if trigger_model is not None:
                # Dynamic trigger using the trained trigger model
                # Create target label
                target_label = torch.tensor([args.target_label]).to(args.device)
                
                # Generate trigger
                _, trigger_clip = trigger_model(x, None, None, None, target_label)
                
                # Apply trigger
                x_poisoned = x + trigger_clip
            else:
                # Basic patch trigger (similar to epoch.py apply_trigger)
                x_poisoned = x.clone()
                x_poisoned[0, -5:, 0] += args.clip_ratio  # Apply basic patch
            
            # Update the dataset with poisoned sample
            x_poisoned = x_poisoned.squeeze(0).cpu()
            
            # Modify the dataset entry
            train_data.feature_df.loc[train_data.all_IDs[idx]] = x_poisoned.numpy()
            
            # Update label to target label
            train_data.labels_df.loc[train_data.all_IDs[idx]] = args.target_label
    
    print(f"Successfully poisoned {num_poison} samples")
    print("=== DATA POISONING COMPLETED ===")
    
    return train_data, poison_indices

def poison_model(model, trigger_model, train_loader, test_loader, args):
    """Train the main model with poisoned data (backdoor injection).
    
    Since data is already poisoned in Step 2, we use clean training here.
    For evaluation, we still use bd_test to measure attack success rate.
    
    Args:
        model: The main classification model to be poisoned
        trigger_model: Trained trigger generator (used for testing ASR)
        train_loader: Training data loader (contains poisoned samples)
        test_loader: Test data loader
        args: Training arguments
    
    Returns:
        Poisoned model and final accuracy
    """
    print("=== STEP 3: MODEL POISONING ===")
    print("Training model on poisoned dataset...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    CA,ASR = [], []
    model_poision_dic = {}
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
    model_poision_dic['CA'] = CA
    model_poision_dic['ASR'] = ASR
    return model_poision_dic

def training_with_poissoned_data(args, trigger_model):
    """Training a network model with poisoned dataset using a dynamic trigger model.
    
    This framework consists of 3 steps:
    1) Trigger Training: Train the dynamic trigger generator
    2) Data Poisoning: Setup for poisoning training data  
    3) Model Poisoning: Train the main model with backdoor

    Args:
        args: Training arguments
        trigger_model: Trigger generator network (Bd_MLP, TimesBA, or None for basic)

    Returns:
        final_asr: Final attack success rate
        model: The poisoned model
    """
    print("\n" + "="*60)
    print("BACKDOOR TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Load data
    full_train_data, full_train_loader = get_data(args, 'train')
    test_data, test_loader = get_data(args, 'test')
    args.seq_len = max(full_train_data.max_seq_len, test_data.max_seq_len)

    # Initialize main model
    model = get_clean_model(args, full_train_data, test_data)
    
    trigger_results = None

    # -------------------------
    # STEP 1: Trigger Model Training
    # -------------------------
    if trigger_model is not None:
        # If trigger_model is a placeholder string, create the actual model
        if trigger_model == "placeholder":
            # Use trigger model from dictionary based on surrogate_type
            if args.surrogate_model in trigger_model_dict:
                print(f"Creating {args.surrogate_type} trigger model...")
                trigger_model = trigger_model_dict[args.surrogate_type](args).to(args.device)
            else:
                raise ValueError(f"Unknown surrogate_type: {args.surrogate_type}. "
                               f"Available options: {list(trigger_model_dict.keys())}")
        
        # Create a surrogate model for trigger training
        surrogate_model = get_clean_model(args, full_train_data, test_data,model_type=args.surrogate_model)
        
        # Optionally warmup the surrogate
        if args.warmup_epochs > 0:
            warmup_surrogate(surrogate_model, full_train_loader, args)
        
        # Train the trigger model
        trigger_results = train_trigger_model(
            trigger_model,
            surrogate_model,
            full_train_loader,
            args
        )
    else:
        print("\n=== STEP 1: SKIPPED (Using basic patch trigger) ===")
    
    # -------------------------
    # STEP 2: Data Poisoning Setup
    # -------------------------
    poisoned_data, poison_indices = poison_data(trigger_model, full_train_data, args)
    
    # Recreate data loader with poisoned data
    poisoned_train_loader = DataLoader(
        poisoned_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=functools.partial(collate_fn, max_len=args.seq_len)
    )

    # Loader containing only poisoned samples for latent separability plotting
    from torch.utils.data import Subset
    poisoned_only_loader = None
    if poison_indices:
        poisoned_subset = Subset(poisoned_data, poison_indices)
        poisoned_only_loader = DataLoader(
            poisoned_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            collate_fn=functools.partial(collate_fn, max_len=args.seq_len),
        )
    
    # -------------------------
    # STEP 3: Model Poisoning
    # -------------------------
    model_poison_dic = poison_model(
        model,
        trigger_model,
        poisoned_train_loader,  # Use poisoned data loader
        test_loader,
        args
    )
    final_asr = model_poison_dic['ASR'][-1]

    # Final evaluation to collect samples for visualization/logging
    final_loss, final_clean_acc, final_asr_eval, sample_cases = bd_test_with_samples(
        model, test_loader, args, trigger_model, max_success=8, max_failure=8
    )

    # Persist logs, curves, example plots, and latent separability
    log_all(
        args=args,
        trigger_results=trigger_results,
        model_poison_dic=model_poison_dic,
        sample_cases=sample_cases,
        model=model,
        test_loader=test_loader,
        poisoned_loader=poisoned_only_loader if 'poisoned_only_loader' in locals() else None,
        trigger_model=trigger_model,
        latent_method=getattr(args, "latent_method", "pca"),
        latent_max_points=getattr(args, "latent_max_points", 2000),
    )
    print("\n" + "="*60)
    print("BACKDOOR TRAINING COMPLETED")
    print(f"Final Attack Success Rate: {final_asr:.4f}")
    print("="*60 + "\n")

    return trigger_results, model_poison_dic

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = args_parser()
    #set_seed(42) # for reproducibility, currently no need to use
    
    # Set device early so models can be loaded properly
    args.device = select_least_used_gpu()
    print(f"Using device: {args.device}")
    train_data, train_loader = get_data(args, flag='train')
    test_data, test_loader = get_data(args, flag='test')
    args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
    args.pred_len = 0
    args.enc_in = train_data.feature_df.shape[1]
    args.num_class = len(train_data.class_names)
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    print('number of classes:', args.num_class)

    if args.mode == "clean":
        acc, model = training_with_clean_data(args)

    elif args.mode in ["basic", "marksman", "dynamic"]: 
        trigger_model = trigger_model_dict[args.Tmodel](args).float().to(args.device)  # Default to None (basic patch mode)
        
        trigger_results, model_poison_dic = training_with_poissoned_data(args, trigger_model)
        
    else:
        raise ValueError("Unknown mode: clean | basic | triggerNet")
