"""Utility for loading pre-trained trigger models with dimension adaptation."""

import torch
from pathlib import Path
from utils.helper_train_test import create_trigger_model


def load_trigger_model_with_adaptation(checkpoint_path, args, train_data=None):
    """Load a pre-trained trigger model and adapt it to current dataset dimensions.
    
    This function handles loading trigger models trained on different datasets
    by creating a new model with current dimensions and attempting to load
    compatible weights.
    
    Args:
        checkpoint_path: Path to the saved trigger model checkpoint
        args: Current training arguments
        train_data: Current training dataset (for inferring dimensions)
        
    Returns:
        Loaded trigger model adapted to current dimensions
        
    Note:
        - If dimensions match exactly, all weights are loaded
        - If dimensions differ, only compatible layers are loaded
        - Incompatible layers are initialized randomly with a warning
        
    Example:
        >>> from utils.load_trigger_model import load_trigger_model_with_adaptation
        >>> trigger_model = load_trigger_model_with_adaptation(
        ...     "Results/exp/trigger_cnn_DatasetA_basic.pth",
        ...     args,
        ...     train_data
        ... )
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract saved metadata
    saved_args = checkpoint.get('args', {})
    saved_model_type = checkpoint.get('model_type', args.Tmodel)
    saved_dataset = checkpoint.get('dataset', 'unknown')
    saved_method = checkpoint.get('method', 'basic')
    
    print(f"\n=== Loading Trigger Model ===")
    print(f"Checkpoint: {Path(checkpoint_path).name}")
    print(f"Trained on: {saved_dataset} with {saved_model_type} using {saved_method}")
    
    # Update current dataset dimensions if train_data provided
    if train_data is not None:
        args.enc_in = train_data.feature_df.shape[1]
        args.num_class = int(train_data.labels_df.nunique().values[0])
    
    current_dataset = Path(args.root_path).name
    print(f"Current dataset: {current_dataset}")
    print(f"Current dimensions: enc_in={args.enc_in}, seq_len={args.seq_len}")
    
    # Check dimension compatibility
    saved_enc_in = saved_args.get('enc_in', args.enc_in)
    saved_seq_len = saved_args.get('seq_len', args.seq_len)
    
    dimensions_match = (saved_enc_in == args.enc_in and saved_seq_len == args.seq_len)
    
    if dimensions_match:
        print("✓ Dimensions match - full weight transfer possible")
    else:
        print(f"⚠ Dimension mismatch:")
        print(f"  Saved: enc_in={saved_enc_in}, seq_len={saved_seq_len}")
        print(f"  Current: enc_in={args.enc_in}, seq_len={args.seq_len}")
        print("  Will attempt partial weight transfer for compatible layers")
    
    # Create new model with current dimensions
    trigger_model = create_trigger_model(args).float().to(args.device)
    
    # Load state dict with size mismatch handling
    model_state_dict = checkpoint['model_state_dict']
    current_state_dict = trigger_model.state_dict()
    
    # Filter and load compatible weights
    compatible_weights = {}
    incompatible_keys = []
    
    for key, param in model_state_dict.items():
        if key in current_state_dict:
            if current_state_dict[key].shape == param.shape:
                compatible_weights[key] = param
            else:
                incompatible_keys.append(
                    f"{key}: saved {param.shape} vs current {current_state_dict[key].shape}"
                )
        else:
            incompatible_keys.append(f"{key}: not found in current model")
    
    # Load compatible weights
    trigger_model.load_state_dict(compatible_weights, strict=False)
    
    # Report loading status
    print(f"\nLoaded {len(compatible_weights)}/{len(model_state_dict)} weight tensors")
    
    if incompatible_keys:
        print(f"\n⚠ {len(incompatible_keys)} incompatible weights (randomly initialized):")
        for key in incompatible_keys[:5]:  # Show first 5
            print(f"  - {key}")
        if len(incompatible_keys) > 5:
            print(f"  ... and {len(incompatible_keys) - 5} more")
        print("\nNote: Incompatible layers will need retraining for optimal performance")
    
    print("=== Trigger Model Loaded Successfully ===\n")
    
    return trigger_model
