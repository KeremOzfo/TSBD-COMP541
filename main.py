"""
Main Entry Point for TimeSeries Backdoor Attack

- For clean training: python main.py --mode clean
- For backdoor training: python main.py --mode basic|marksman|dynamic
- For testing pre-trained trigger: python main.py --mode test --trigger_model_path <path>
"""

import os
import torch
import numpy as np
import random

from parameters import args_parser
from utils.tools import load_and_override_params
from train import (
    get_data,
    training_with_clean_data,
    train_trigger_model,
    poison_data,
    poison_model
)
from utils.helper_train_test import create_trigger_model
from test import test_pretrained_trigger
import functools
from torch.utils.data import DataLoader
from utils.helper_train_test import reconfigure_model_for_data
from data_provider.uea import collate_fn
from utils.exp_logging import log_final_test_epoch
from utils.helper_gpu import select_least_used_gpu
from utils.auto_arch import apply_auto_bd_arch, apply_auto_batch_size, DatasetInfo


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = args_parser()
    #args = load_and_override_params(args)
    if not torch.cuda.is_available(): 
        args.device = torch.device("cpu")
    else:
        args.device = select_least_used_gpu()
    print(f"Using device: {args.device}")
    train_data, train_loader = get_data(args, flag='train')
    test_data, test_loader = get_data(args, flag='test')
    
    # Update seq_len to match actual data if not already set correctly
    actual_max_len = max(train_data.max_seq_len, test_data.max_seq_len)
    if args.seq_len != actual_max_len:
        print(f"[Warning] Updating seq_len from {args.seq_len} to {actual_max_len} to match dataset")
        args.seq_len = actual_max_len
        # Recreate loaders with correct max_len
        train_data, train_loader = get_data(args, flag='train')
        test_data, test_loader = get_data(args, flag='test')
    
    args.enc_in = train_data.feature_df.shape[1]
    args.num_class = len(train_data.class_names)

    # Auto-select trigger architecture based on dataset complexity
    if getattr(args, "auto_bd_arch", False):
        dataset_name = os.path.basename(args.root_path.rstrip("/\\"))
        fallback_info = DatasetInfo(
            name=dataset_name,
            seq_len=int(args.seq_len),
            num_variates=int(args.enc_in),
            num_classes=int(args.num_class),
            num_train=int(len(train_data)),
            num_test=int(len(test_data)),
        )
        apply_auto_bd_arch(args, dataset_name, args.dataset_info_path, fallback=fallback_info)

    if getattr(args, "auto_batch_size", False):
        dataset_name = os.path.basename(args.root_path.rstrip("/\\"))
        fallback_info = DatasetInfo(
            name=dataset_name,
            seq_len=int(args.seq_len),
            num_variates=int(args.enc_in),
            num_classes=int(args.num_class),
            num_train=int(len(train_data)),
            num_test=int(len(test_data)),
        )
        apply_auto_batch_size(args, dataset_name, args.dataset_info_path, fallback=fallback_info)
    
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    print(f"Number of classes: {args.num_class}")
    print(f"Trigger Training Method: {args.method}")
    print(f"Backdoor Type: {args.bd_type}")

    
    if args.train_mode == "clean":
        acc, model = training_with_clean_data(args)
        exit(0)
    elif args.train_mode == "backdoor":
   
        if args.use_pretrained_trigger == True:  
            trigger_model = torch.load(args.trigger_model_path).to(args.device)
            trigger_results = None
            mask_model = None
        else:
            trigger_model = create_trigger_model(args).float().to(args.device)
            surrogate_model = reconfigure_model_for_data(args,train_data,test_data,model_type=args.surrogate_model)
            
            # Use mask-based training if method is inputaware_masking
            if args.method == "inputaware_masking":
                from train import train_trigger_model_with_mask
                trigger_results = train_trigger_model_with_mask(trigger_model, surrogate_model, train_loader, args, train_data)
                mask_model = trigger_results.get('mask_model', None)
            elif args.method == "defeat":
                from train import train_defeat
                trigger_results = train_defeat(trigger_model, surrogate_model, train_loader, args, train_data)
                mask_model = None
                # Store aux_logits for test phase
                aux_logits = trigger_results.get('aux_logits', None)
            else:
                trigger_results = train_trigger_model(trigger_model, surrogate_model, train_loader, args, train_data)
                mask_model = None
                aux_logits = None

        poisoned_dataset, poison_indices = poison_data(trigger_model, train_data, args, mask_model)
        poisoned_train_loader = DataLoader(poisoned_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,drop_last=False,collate_fn=functools.partial(collate_fn, max_len=args.seq_len))
        target_model = reconfigure_model_for_data(args,train_data,test_data)
        model_poison_dic = poison_model(target_model,trigger_model,poisoned_train_loader,test_loader,args)
        
        # Final comprehensive test epoch with all visualizations
        print("\n" + "="*70)
        print("Starting final comprehensive test epoch...")
        print("="*70)
        
        # Import and run bd_test_with_samples to collect sample cases
        from test import bd_test_with_samples
        test_loss, clean_acc, asr, sample_cases = bd_test_with_samples(
            model=target_model,
            loader=test_loader,
            args=args,
            trigger_model=trigger_model,
            mask_model=mask_model if args.method == "inputaware_masking" else None,
            max_success=2,
            max_failure=1,
        )
        print(f"Final CA: {clean_acc*100:.2f}%, ASR: {asr*100:.2f}%")
        
        # Log all results
        exp_dir = log_final_test_epoch(
            model=target_model,
            trigger_model=trigger_model,
            train_loader=train_loader,
            test_loader=test_loader,
            poisoned_loader=poisoned_train_loader,
            args=args,
            sample_cases=sample_cases,  # Pass pre-collected samples
            poison_indices=poison_indices,  # Pass poisoned sample indices
            trigger_results=trigger_results,
            model_poison_dic=model_poison_dic,
            save_dir="Results",
            run_bd_test=False,  # Already ran it above
        )
        print(f"\nAll results saved to: {exp_dir}")


    elif args.train_mode == "test":
        # Test mode: load pre-trained trigger, poison data, train victim, evaluate
        trigger_path = getattr(args, 'trigger_model_path', None)
        
        if trigger_path is None:
            raise ValueError(
                "Test mode requires --trigger_model_path argument.\n"
                "Usage: python main.py --mode test --trigger_model_path <path>"
            )
        
        results = test_pretrained_trigger(args, trigger_path)


    else:
        raise ValueError(
            f"Unknown method: {args.method}. "
            "Choose from: clean | backdoor | test"
        )
