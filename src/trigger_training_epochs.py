import torch
import torch.nn as nn
import numpy as np
from src.methods.vanilla_trigger_training import epoch_vanilla_trigger_training
from src.methods.marksman import epoch_marksman
from src.methods.diversity_legacy import epoch_diversity
from src.methods.epoch_ultimate import epoch_combined
from src.methods.frequency_guided import epoch_frequency_guided
from src.methods.input_aware_dynamic_trigger_training import epoch_marksman_input_aware
from src.methods.pure_inputaware import epoch_pure_input_aware
from src.methods.inputaware_masking import epoch_inputaware_masking, train_mask_epoch
from src.methods.defeat import epoch_defeat

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

def trigger_train_epoch(trigger_model, surrogate_model, loader, args, optimizer_trigger, optimizer_surrogate, trigger_model_prev=None, loader2=None): 
    """Train trigger generator for one epoch using the selected method.
    
    Args:
        trigger_model: Trigger generator network
        surrogate_model: Surrogate classifier
        loader: Primary data loader
        args: Training arguments (includes args.method for method selection)
        optimizer_trigger: Optimizer for trigger generator
        optimizer_surrogate: Optimizer for surrogate classifier
        trigger_model_prev: Previous trigger model (for ultimate/inputaware methods)
        loader2: Secondary data loader for diversity/cross-trigger (for diversity/ultimate/inputaware)
    
    Returns:
        Method-specific training results (loss, clean_acc, bd_acc, ...)
    """ 
    
    if args.method == 'marksman':
        return epoch_marksman(
            trigger_model=trigger_model,
            surr_model=surrogate_model,
            loader=loader,
            args=args,
            opt_cls=optimizer_surrogate,
            opt_trig=optimizer_trigger,
            train=True,
        )
    elif args.method == 'basic':
        return epoch_vanilla_trigger_training(
            trigger_model=trigger_model,
            surrogate_model=surrogate_model,
            loader=loader,
            args=args,
            optimizer_trigger=optimizer_trigger,
            optimizer_surrogate=optimizer_surrogate,
            train=True,
        )

    elif args.method == 'diversity':
        return epoch_diversity(
            bd_model=trigger_model,
            surr_model=surrogate_model,
            loader1=loader,
            args=args,
            loader2=loader2,
            opt=optimizer_trigger,
            opt2=optimizer_surrogate,
            train=True,
        )

    elif args.method == 'ultimate':
        return epoch_combined(
            trigger_model=trigger_model,
            trigger_model_prev=trigger_model_prev,
            surrogate_model=surrogate_model,
            loader1=loader,
            args=args,
            loader2=loader2,
            opt_trig=optimizer_trigger,
            opt_class=optimizer_surrogate,
            train=True,
            freq_target=None,
        )
    
    elif args.method == "frequency":
        return epoch_frequency_guided(
            bd_model=trigger_model,
            surr_model=surrogate_model,
            loader=loader,
            args=args,
            opt=optimizer_trigger,
            train=True,
        )
    elif args.method == "inputaware":
        return epoch_marksman_input_aware(
            bd_model=trigger_model,
            bd_model_prev=trigger_model_prev,
            surr_model=surrogate_model,
            loader1=loader,
            args=args,
            loader2=loader2,
            opt_trig=optimizer_trigger,
            opt_class=optimizer_surrogate,
            train=True,
        )
    
    elif args.method == "pureinputaware":
        return epoch_pure_input_aware(
            bd_model=trigger_model,
            bd_model_prev=trigger_model_prev,
            surr_model=surrogate_model,
            loader1=loader,
            args=args,
            loader2=loader2,
            opt_trig=optimizer_trigger,
            opt_class=optimizer_surrogate,
            train=True,
        )
    
    elif args.method == "inputaware_masking":
        # This method requires mask_model to be passed separately
        # It should be called from train_trigger_model which handles mask pre-training
        raise RuntimeError("inputaware_masking should be called via train_trigger_model_with_mask")
    
    elif args.method == "defeat":
        # DEFEAT uses its own training procedure (not epoch-based)
        # Should be called from train_defeat in train.py
        raise RuntimeError("defeat method should be called via train_defeat")

    else:
        raise NotImplementedError(f"Unknown training mode: {args.method}")


def trigger_eval_epoch(trigger_model, surrogate_model, loader, args, trigger_model_prev=None):
    """Evaluate trigger model without training (eval mode).
    
    This function dispatches to the appropriate method's epoch function
    with optimizers set to None (or train=False) to run in evaluation mode.
    
    Args:
        trigger_model: Trigger generator network
        surrogate_model: Surrogate classifier
        loader: Data loader
        args: Arguments (includes args.method for method selection)
        trigger_model_prev: Previous trigger model (for methods that need it)
    
    Returns:
        Method-specific evaluation results (loss, accuracies, etc.)
    """
    if args.method == 'marksman':
        return epoch_marksman(
            trigger_model=trigger_model,
            surr_model=surrogate_model,
            loader=loader,
            args=args,
            opt_cls=None,
            opt_trig=None,
            train=False,
        )
    elif args.method == 'basic':
        return epoch_vanilla_trigger_training(
            trigger_model=trigger_model,
            surrogate_model=surrogate_model,
            loader=loader,
            args=args,
            optimizer_trigger=None,
            optimizer_surrogate=None,
            train=False,
        )
    elif args.method == 'diversity':
        return epoch_diversity(
            bd_model=trigger_model,
            surr_model=surrogate_model,
            loader1=loader,
            args=args,
            loader2=None,
            opt=None,
            opt2=None,
            train=False,
        )
    elif args.method == 'ultimate':
        return epoch_combined(
            trigger_model=trigger_model,
            trigger_model_prev=trigger_model_prev,
            surrogate_model=surrogate_model,
            loader1=loader,
            args=args,
            loader2=None,
            opt_trig=None,
            opt_class=None,
            train=False,
            freq_target=None,
        )
    elif args.method == 'frequency':
        return epoch_frequency_guided(
            bd_model=trigger_model,
            surr_model=surrogate_model,
            loader=loader,
            args=args,
            opt=None,
            train=False,
        )
    elif args.method == 'inputaware':
        return epoch_marksman_input_aware(
            bd_model=trigger_model,
            bd_model_prev=trigger_model_prev,
            surr_model=surrogate_model,
            loader1=loader,
            args=args,
            loader2=None,
            opt_trig=None,
            opt_class=None,
            train=False,
        )
    else:
        raise NotImplementedError(f"Unknown eval mode: {args.method}")

