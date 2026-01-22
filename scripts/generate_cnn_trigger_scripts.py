"""
CNN-based Trigger Network Parameter Exploration Script Generator

This script generates comprehensive training scripts for CNN-based trigger models:
- Trigger models: ccnn, ccnn_cae
- Training methods: vanilla, marksman, diversity, ultimate, frequency, inputaware, pureinputaware
- Optimizer configurations (type, lr, weight decay, momentum)
- Training dynamics (warmup, epochs, batch size)
- Regularization (L2 penalty)
- Poisoning strategies (ratio, silent poisoning)

Note: CNN models don't require d_model_bd, d_ff_bd, e_layers_bd, n_heads_bd parameters,
so the parameter space is simpler than transformer-based models.
"""
import csv
import os

# Read dataset info
datasets = []
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'dataset_info_subset.csv')
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        datasets.append({
            'name': row['dataset'],
            'seq_len': int(row['seq_len']),
            'num_variates': int(row['num_variates']),
            'num_classes': int(row['num_classes']),
            'num_train': int(row['num_train']),
            'num_test': int(row['num_test'])
        })

# CNN-based trigger models
TRIGGER_MODELS = ['ccnn', 'ccnn_cae']

# Main classifier models to explore
MAIN_MODELS = ['TimesNet','TCN','lstm']

# All methods except defeat and basic
#TRAINING_METHODS = ['vanilla', 'marksman', 'diversity', 'ultimate', 'frequency', 'inputaware', 'pureinputaware']
#TRAINING_METHODS = ['vanilla', 'marksman'] # check vanilla and marksman first
TRAINING_METHODS = ['pureinputaware','inputaware_masking']  # for quick testing
# ==================== OPTIMIZER CONFIGURATIONS ====================
OPTIMIZER_CONFIGS = {
    # Standard configurations
    'adam_standard': {
        'trigger_opt': 'adam',
        'trigger_lr': 1e-3,
        'trigger_weight_decay': 0.0,
        'surrogate_opt': 'adam',
        'surrogate_lr': 1e-3,
        'surrogate_weight_decay': 0.0,
        'surrogate_L2_penalty': 0.0,
        'description': 'Standard Adam - baseline configuration'
    },
    'adam_regularized': {
        'trigger_opt': 'adam',
        'trigger_lr': 5e-4,
        'trigger_weight_decay': 1e-5,
        'surrogate_opt': 'adam',
        'surrogate_lr': 5e-4,
        'surrogate_weight_decay': 1e-4,
        'surrogate_L2_penalty': 1e-3,
        'description': 'Regularized Adam - prevents overfitting'
    },
    'adam_aggressive': {
        'trigger_opt': 'adam',
        'trigger_lr': 5e-3,
        'trigger_weight_decay': 0.0,
        'surrogate_opt': 'adam',
        'surrogate_lr': 1e-2,
        'surrogate_weight_decay': 0.0,
        'surrogate_L2_penalty': 0.0,
        'description': 'Aggressive Adam - fast convergence'
    },
    
    # AdamW configurations (better weight decay handling)
    'adamw_standard': {
        'trigger_opt': 'adamw',
        'trigger_lr': 1e-3,
        'trigger_weight_decay': 1e-2,
        'surrogate_opt': 'adamw',
        'surrogate_lr': 1e-3,
        'surrogate_weight_decay': 1e-2,
        'surrogate_L2_penalty': 0.0,
        'description': 'AdamW - decoupled weight decay'
    },
    'adamw_light': {
        'trigger_opt': 'adamw',
        'trigger_lr': 1e-3,
        'trigger_weight_decay': 1e-3,
        'surrogate_opt': 'adamw',
        'surrogate_lr': 1e-3,
        'surrogate_weight_decay': 1e-3,
        'surrogate_L2_penalty': 0.0,
        'description': 'AdamW light regularization'
    },
    'adamw_heavy': {
        'trigger_opt': 'adamw',
        'trigger_lr': 5e-4,
        'trigger_weight_decay': 5e-2,
        'surrogate_opt': 'adamw',
        'surrogate_lr': 5e-4,
        'surrogate_weight_decay': 5e-2,
        'surrogate_L2_penalty': 1e-2,
        'description': 'AdamW heavy regularization - for large models'
    },
    
    # SGD configurations (momentum-based)
    'sgd_momentum': {
        'trigger_opt': 'sgd',
        'trigger_lr': 1e-2,
        'trigger_weight_decay': 5e-4,
        'surrogate_opt': 'sgd',
        'surrogate_lr': 1e-2,
        'surrogate_weight_decay': 5e-4,
        'surrogate_L2_penalty': 0.0,
        'description': 'SGD with momentum - stable gradients'
    },
    'sgd_aggressive': {
        'trigger_opt': 'sgd',
        'trigger_lr': 5e-2,
        'trigger_weight_decay': 1e-3,
        'surrogate_opt': 'sgd',
        'surrogate_lr': 5e-2,
        'surrogate_weight_decay': 1e-3,
        'surrogate_L2_penalty': 0.0,
        'description': 'Aggressive SGD - fast but may be unstable'
    },
    
    # Mixed configurations
    'mixed_adam_sgd': {
        'trigger_opt': 'adam',
        'trigger_lr': 1e-3,
        'trigger_weight_decay': 0.0,
        'surrogate_opt': 'sgd',
        'surrogate_lr': 1e-2,
        'surrogate_weight_decay': 5e-4,
        'surrogate_L2_penalty': 0.0,
        'description': 'Adam for trigger, SGD for surrogate'
    },
    'mixed_sgd_adam': {
        'trigger_opt': 'sgd',
        'trigger_lr': 1e-2,
        'trigger_weight_decay': 5e-4,
        'surrogate_opt': 'adam',
        'surrogate_lr': 1e-3,
        'surrogate_weight_decay': 0.0,
        'surrogate_L2_penalty': 0.0,
        'description': 'SGD for trigger, Adam for surrogate'
    }
}

OPTIMIZER_CONFIGS_TOP3 = {
    # RANK 1: Best overall - Adam for trigger, SGD for surrogate (Mean CA: 0.8162)
    'adam_sgd_best': {
        'trigger_opt': 'adam',
        'trigger_lr': 1e-3,
        'trigger_weight_decay': 0.0,
        'surrogate_opt': 'sgd',
        'surrogate_lr': 1e-2,
        'surrogate_weight_decay': 5e-4,
        'surrogate_L2_penalty': 0.0,
        'description': 'RANK 1: Best performing - Adam/SGD (CA: 0.8162)'
    },
    
    # RANK 2: Best single optimizer - SGD for both (Mean CA: 0.8113)
    'sgd_sgd_best': {
        'trigger_opt': 'sgd',
        'trigger_lr': 5e-2,
        'trigger_weight_decay': 1e-3,
        'surrogate_opt': 'sgd',
        'surrogate_lr': 5e-2,
        'surrogate_weight_decay': 1e-3,
        'surrogate_L2_penalty': 0.0,
        'description': 'RANK 2: Best single optimizer - SGD/SGD (CA: 0.8113)'
    },
    
    # RANK 3: Balanced - SGD for trigger, Adam for surrogate (Mean CA: 0.8102)
    'sgd_adam_best': {
        'trigger_opt': 'sgd',
        'trigger_lr': 1e-2,
        'trigger_weight_decay': 5e-4,
        'surrogate_opt': 'adam',
        'surrogate_lr': 1e-3,
        'surrogate_weight_decay': 0.0,
        'surrogate_L2_penalty': 0.0,
        'description': 'RANK 3: Balanced performance - SGD/Adam (CA: 0.8102)'
    }
}

# Use TOP 3 configs for new experiments
OPTIMIZER_CONFIGS = OPTIMIZER_CONFIGS_TOP3

# ==================== METHOD-SPECIFIC HYPERPARAMETERS ====================
METHOD_HYPERPARAMS = {
    'vanilla': [
        {'name': 'default', 'params': {}},
    ],
    'marksman': [
        {'name': 'balanced', 'params': {'marksman_alpha': 0.5, 'marksman_beta': 0.0, 'marksman_update_T': 1}},
        {'name': 'high_alpha', 'params': {'marksman_alpha': 0.7, 'marksman_beta': 0.0, 'marksman_update_T': 1}},
        {'name': 'low_alpha', 'params': {'marksman_alpha': 0.3, 'marksman_beta': 0.0, 'marksman_update_T': 1}},
        {'name': 'with_penalty', 'params': {'marksman_alpha': 0.5, 'marksman_beta': 0.001, 'marksman_update_T': 1}},
        {'name': 'high_penalty', 'params': {'marksman_alpha': 0.5, 'marksman_beta': 0.01, 'marksman_update_T': 1}},
        {'name': 'slow_update', 'params': {'marksman_alpha': 0.5, 'marksman_beta': 0.0, 'marksman_update_T': 3}},
        {'name': 'aggressive', 'params': {'marksman_alpha': 0.8, 'marksman_beta': 0.001, 'marksman_update_T': 1}},
    ],
    'diversity': [
        {'name': 'low_div', 'params': {'div_reg': 0.5, 'poisoning_ratio_train': 0.1}},
        {'name': 'standard', 'params': {'div_reg': 1.0, 'poisoning_ratio_train': 0.1}},
        {'name': 'high_div', 'params': {'div_reg': 2.0, 'poisoning_ratio_train': 0.1}},
        {'name': 'very_high_div', 'params': {'div_reg': 5.0, 'poisoning_ratio_train': 0.1}},
        {'name': 'high_poison', 'params': {'div_reg': 1.0, 'poisoning_ratio_train': 0.2}},
        {'name': 'low_poison', 'params': {'div_reg': 1.0, 'poisoning_ratio_train': 0.05}},
    ],
    'frequency': [
        {'name': 'low_freq', 'params': {'lambda_freq': 0.5, 'freq_lambda': 0.05}},
        {'name': 'standard', 'params': {'lambda_freq': 1.0, 'freq_lambda': 0.05}},
        {'name': 'high_freq', 'params': {'lambda_freq': 2.0, 'freq_lambda': 0.05}},
        {'name': 'very_high_freq', 'params': {'lambda_freq': 5.0, 'freq_lambda': 0.05}},
        {'name': 'high_perturbation', 'params': {'lambda_freq': 1.0, 'freq_lambda': 0.1}},
        {'name': 'low_perturbation', 'params': {'lambda_freq': 1.0, 'freq_lambda': 0.01}},
    ],
    'inputaware': [
        {'name': 'balanced', 'params': {'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 1.0}},
        {'name': 'high_attack', 'params': {'p_attack': 0.7, 'p_cross': 0.1, 'lambda_cross': 1.0}},
        {'name': 'low_attack', 'params': {'p_attack': 0.3, 'p_cross': 0.1, 'lambda_cross': 1.0}},
        {'name': 'high_cross', 'params': {'p_attack': 0.5, 'p_cross': 0.2, 'lambda_cross': 1.0}},
        {'name': 'low_cross', 'params': {'p_attack': 0.5, 'p_cross': 0.05, 'lambda_cross': 1.0}},
        {'name': 'strong_cross_loss', 'params': {'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 2.0}},
        {'name': 'weak_cross_loss', 'params': {'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 0.5}},
        {'name': 'aggressive', 'params': {'p_attack': 0.7, 'p_cross': 0.2, 'lambda_cross': 2.0}},
    ],
    'pureinputaware': [
        {'name': 'balanced', 'params': {'p_attack': 0.5, 'lambda_div': 1.0, 'lambda_cross': 1.0}},
        {'name': 'high_attack', 'params': {'p_attack': 0.7, 'lambda_div': 1.0, 'lambda_cross': 1.0}},
        {'name': 'low_attack', 'params': {'p_attack': 0.3, 'lambda_div': 1.0, 'lambda_cross': 1.0}},
        {'name': 'high_diversity', 'params': {'p_attack': 0.5, 'lambda_div': 2.0, 'lambda_cross': 1.0}},
        {'name': 'high_cross', 'params': {'p_attack': 0.5, 'lambda_div': 1.0, 'lambda_cross': 2.0}},
        {'name': 'aggressive', 'params': {'p_attack': 0.7, 'lambda_div': 2.0, 'lambda_cross': 2.0}},
    ],
    'ultimate': [
        {'name': 'balanced', 'params': {'lambda_freq': 1.0, 'lambda_div': 1.0, 'lambda_reg': 1e-3, 'lambda_cross': 1.0, 'p_attack': 0.5, 'p_cross': 0.1}},
        {'name': 'freq_focused', 'params': {'lambda_freq': 3.0, 'lambda_div': 0.5, 'lambda_reg': 1e-3, 'lambda_cross': 0.5, 'p_attack': 0.5, 'p_cross': 0.1}},
        {'name': 'div_focused', 'params': {'lambda_freq': 0.5, 'lambda_div': 3.0, 'lambda_reg': 1e-3, 'lambda_cross': 0.5, 'p_attack': 0.5, 'p_cross': 0.1}},
        {'name': 'cross_focused', 'params': {'lambda_freq': 0.5, 'lambda_div': 0.5, 'lambda_reg': 1e-3, 'lambda_cross': 3.0, 'p_attack': 0.5, 'p_cross': 0.2}},
        {'name': 'high_reg', 'params': {'lambda_freq': 1.0, 'lambda_div': 1.0, 'lambda_reg': 1e-2, 'lambda_cross': 1.0, 'p_attack': 0.5, 'p_cross': 0.1}},
        {'name': 'aggressive', 'params': {'lambda_freq': 2.0, 'lambda_div': 2.0, 'lambda_reg': 1e-3, 'lambda_cross': 2.0, 'p_attack': 0.7, 'p_cross': 0.15}},
    ],
    'inputaware_masking': [
        # Balanced baseline configurations
        {'name': 'balanced', 'params': {
            'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 1.0,
            'mask_density': 0.05, 'mask_lr': 1e-3, 'mask_weight_decay': 0.0,
            'mask_pretrain_epochs': 20
        }},
        
        # Vary attack probability
        {'name': 'high_attack', 'params': {
            'p_attack': 0.7, 'p_cross': 0.1, 'lambda_cross': 1.0,
            'mask_density': 0.05, 'mask_lr': 1e-3, 'mask_weight_decay': 0.0,
            'mask_pretrain_epochs': 20
        }},
        {'name': 'low_attack', 'params': {
            'p_attack': 0.3, 'p_cross': 0.1, 'lambda_cross': 1.0,
            'mask_density': 0.05, 'mask_lr': 1e-3, 'mask_weight_decay': 0.0,
            'mask_pretrain_epochs': 20
        }},
        
        # Vary cross-entropy loss
        {'name': 'high_cross', 'params': {
            'p_attack': 0.5, 'p_cross': 0.2, 'lambda_cross': 1.0,
            'mask_density': 0.05, 'mask_lr': 1e-3, 'mask_weight_decay': 0.0,
            'mask_pretrain_epochs': 20
        }},
        {'name': 'strong_cross_loss', 'params': {
            'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 2.0,
            'mask_density': 0.05, 'mask_lr': 1e-3, 'mask_weight_decay': 0.0,
            'mask_pretrain_epochs': 20
        }},
        {'name': 'weak_cross_loss', 'params': {
            'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 0.5,
            'mask_density': 0.05, 'mask_lr': 1e-3, 'mask_weight_decay': 0.0,
            'mask_pretrain_epochs': 20
        }},
        
        # Vary mask density (sparsity control)
        {'name': 'sparse_mask', 'params': {
            'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 1.0,
            'mask_density': 0.02, 'mask_lr': 1e-3, 'mask_weight_decay': 0.0,
            'mask_pretrain_epochs': 20
        }},
        {'name': 'dense_mask', 'params': {
            'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 1.0,
            'mask_density': 0.1, 'mask_lr': 1e-3, 'mask_weight_decay': 0.0,
            'mask_pretrain_epochs': 20
        }},
        {'name': 'very_dense_mask', 'params': {
            'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 1.0,
            'mask_density': 0.2, 'mask_lr': 1e-3, 'mask_weight_decay': 0.0,
            'mask_pretrain_epochs': 20
        }},
        
        # Vary mask learning rate
        {'name': 'low_mask_lr', 'params': {
            'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 1.0,
            'mask_density': 0.05, 'mask_lr': 5e-4, 'mask_weight_decay': 0.0,
            'mask_pretrain_epochs': 20
        }},
        {'name': 'high_mask_lr', 'params': {
            'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 1.0,
            'mask_density': 0.05, 'mask_lr': 5e-3, 'mask_weight_decay': 0.0,
            'mask_pretrain_epochs': 20
        }},
        {'name': 'very_high_mask_lr', 'params': {
            'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 1.0,
            'mask_density': 0.05, 'mask_lr': 1e-2, 'mask_weight_decay': 0.0,
            'mask_pretrain_epochs': 20
        }},
        
        # Vary mask weight decay (regularization)
        {'name': 'light_mask_reg', 'params': {
            'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 1.0,
            'mask_density': 0.05, 'mask_lr': 1e-3, 'mask_weight_decay': 1e-4,
            'mask_pretrain_epochs': 20
        }},
        {'name': 'heavy_mask_reg', 'params': {
            'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 1.0,
            'mask_density': 0.05, 'mask_lr': 1e-3, 'mask_weight_decay': 1e-3,
            'mask_pretrain_epochs': 20
        }},
        
        # Vary mask pretrain epochs
        {'name': 'short_pretrain', 'params': {
            'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 1.0,
            'mask_density': 0.05, 'mask_lr': 1e-3, 'mask_weight_decay': 0.0,
            'mask_pretrain_epochs': 10
        }},
        {'name': 'long_pretrain', 'params': {
            'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 1.0,
            'mask_density': 0.05, 'mask_lr': 1e-3, 'mask_weight_decay': 0.0,
            'mask_pretrain_epochs': 40
        }},
        {'name': 'very_long_pretrain', 'params': {
            'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 1.0,
            'mask_density': 0.05, 'mask_lr': 1e-3, 'mask_weight_decay': 0.0,
            'mask_pretrain_epochs': 60
        }},
        
        # Combined aggressive configurations
        {'name': 'aggressive', 'params': {
            'p_attack': 0.7, 'p_cross': 0.2, 'lambda_cross': 2.0,
            'mask_density': 0.1, 'mask_lr': 5e-3, 'mask_weight_decay': 0.0,
            'mask_pretrain_epochs': 40
        }},
        {'name': 'aggressive_sparse', 'params': {
            'p_attack': 0.7, 'p_cross': 0.2, 'lambda_cross': 2.0,
            'mask_density': 0.02, 'mask_lr': 5e-3, 'mask_weight_decay': 1e-4,
            'mask_pretrain_epochs': 40
        }},
        
        # Conservative configurations
        {'name': 'conservative', 'params': {
            'p_attack': 0.3, 'p_cross': 0.05, 'lambda_cross': 0.5,
            'mask_density': 0.02, 'mask_lr': 5e-4, 'mask_weight_decay': 1e-3,
            'mask_pretrain_epochs': 10
        }},
        
        # Optimal combination explorations
        {'name': 'dense_slow_long', 'params': {
            'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 1.0,
            'mask_density': 0.1, 'mask_lr': 5e-4, 'mask_weight_decay': 1e-4,
            'mask_pretrain_epochs': 40
        }},
        {'name': 'sparse_fast_short', 'params': {
            'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 1.0,
            'mask_density': 0.02, 'mask_lr': 5e-3, 'mask_weight_decay': 0.0,
            'mask_pretrain_epochs': 10
        }},
    ]
}

# ==================== TRAINING DYNAMICS ====================
WARMUP_EPOCHS = [0, 3]  # Different warmup strategies
BATCH_SIZES = [128]  # Different batch sizes affect gradient quality

# ==================== POISONING STRATEGIES ====================
POISONING_CONFIGS = [
    {'poisoning_ratio': 0.1, 'use_silent_poisoning': False, 'name': 'normal_10pct'},
]

def get_trigger_epochs(num_classes):
    """Calculate trigger training epochs based on number of classes.
    
    For conditional backdoor (all2all), more classes require more training
    to learn class-specific triggers effectively.
    
    Args:
        num_classes: Number of classes in the dataset
    
    Returns:
        int: Number of trigger training epochs (15 × num_classes)
    """
    return 15 * num_classes

def generate_script_command(dataset, tmodel, model, method, opt_config, method_params, 
                            warmup, trigger_epochs, batch_size, poison_config):
    """Generate a single training command with all parameters."""
    name = dataset['name']
    seq_len = dataset['seq_len']
    
    # Base command (CNN models don't need patch_len, stride, or architecture params)
    cmd = f"python -u main.py --train_mode backdoor --method {method} --Tmodel {tmodel} --model {model} --bd_type all2all"
    cmd += f" --root_path ./dataset/{name}"
    cmd += f" --seq_len {seq_len}"
    cmd += f" --batch_size {batch_size}"
    cmd += f" --trigger_epochs {trigger_epochs}"
    cmd += f" --bd_train_epochs 30"
    cmd += f" --warmup_epochs {warmup}"
    cmd += f" --target_label 0"
    
    # Optimizer parameters
    cmd += f" --trigger_opt {opt_config['trigger_opt']}"
    cmd += f" --trigger_lr {opt_config['trigger_lr']}"
    cmd += f" --trigger_weight_decay {opt_config['trigger_weight_decay']}"
    cmd += f" --surrogate_opt {opt_config['surrogate_opt']}"
    cmd += f" --surrogate_lr {opt_config['surrogate_lr']}"
    cmd += f" --surrogate_weight_decay {opt_config['surrogate_weight_decay']}"
    cmd += f" --surrogate_L2_penalty {opt_config['surrogate_L2_penalty']}"
    
    # Poisoning configuration
    cmd += f" --poisoning_ratio {poison_config['poisoning_ratio']}"
    if poison_config['use_silent_poisoning']:
        # Only add the flag if silent poisoning is enabled (action='store_true')
        cmd += " --use_silent_poisoning"
        cmd += f" --lambda_ratio {poison_config['lambda_ratio']}"
    
    # Method-specific parameters
    for key, value in method_params.items():
        cmd += f" --{key} {value}"
    
    return cmd + "\n"

def main():
    """Generate comprehensive CNN trigger exploration scripts - ONE DATASET AT A TIME."""
    base_output_dir = os.path.join(script_dir, 'cnn_per_dataset')
    os.makedirs(base_output_dir, exist_ok=True)
    
    total_experiments_all = 0
    dataset_summaries = []
    
    # Generate scripts for EACH dataset separately
    for ds in datasets:
        dataset_name = ds['name']
        dataset_dir = os.path.join(base_output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Generating scripts for dataset: {dataset_name}")
        print(f"{'='*70}")
        
        dataset_total = 0
        method_counts = {}
        
        # Generate scripts organized by method for this dataset
        for method in TRAINING_METHODS:
            method_script = f"#!/bin/bash\n"
            method_script += f"# Dataset: {dataset_name}\n"
            method_script += f"# Method: {method}\n"
            method_script += f"# CNN-based Trigger Models (ccnn, ccnn_cae)\n"
            method_script += f"# Seq Len: {ds['seq_len']}, Variates: {ds['num_variates']}, Classes: {ds['num_classes']}\n\n"
            
            method_count = 0
            
            for tmodel in TRIGGER_MODELS:
                method_script += f"\n# ========== TRIGGER MODEL: {tmodel} ==========\n\n"
                
                # Calculate trigger epochs based on number of classes
                trigger_epochs = get_trigger_epochs(ds['num_classes'])
                
                # Iterate through main models
                for model in MAIN_MODELS:
                    method_script += f"# Main Model: {model}\n"
                    
                    # Iterate through configurations (no architecture params for CNN)
                    for opt_name, opt_config in OPTIMIZER_CONFIGS.items():
                        for method_config in METHOD_HYPERPARAMS[method]:
                            for warmup in WARMUP_EPOCHS:
                                for batch_size in BATCH_SIZES:
                                    for poison_config in POISONING_CONFIGS:
                                        
                                        # Generate comment
                                        comment = f"# {tmodel} | {model} | opt={opt_name} | "
                                        comment += f"cfg={method_config['name']} | warmup={warmup} | "
                                        comment += f"epochs={trigger_epochs} | bs={batch_size} | poison={poison_config['name']}\n"
                                        
                                        method_script += comment
                                        
                                        # Generate command
                                        cmd = generate_script_command(
                                            ds, tmodel, model, method, opt_config,
                                            method_config['params'], warmup, trigger_epochs,
                                            batch_size, poison_config
                                        )
                                        
                                        method_script += cmd + "\n"
                                        
                                        method_count += 1
                                        dataset_total += 1
            
            # Save method script for this dataset
            method_file = os.path.join(dataset_dir, f'{method}.sh')
            with open(method_file, 'w', newline='\n') as f:
                f.write(method_script)
            
            method_counts[method] = method_count
            print(f"  ✓ {method}: {method_count} experiments")
        
        # Generate master script for this dataset
        master_script = f"#!/bin/bash\n"
        master_script += f"# CNN TRIGGER PARAMETER EXPLORATION\n"
        master_script += f"# Dataset: {dataset_name}\n"
        master_script += f"# Total Experiments: {dataset_total}\n\n"
        
        for method in TRAINING_METHODS:
            master_script += f"echo 'Running {method} experiments...'\n"
            master_script += f"bash {method}.sh\n\n"
        
        master_file = os.path.join(dataset_dir, 'run_all_methods.sh')
        with open(master_file, 'w', newline='\n') as f:
            f.write(master_script)
        
        # Generate dataset-specific README
        readme = f"# CNN Trigger Exploration: {dataset_name}\n\n"
        readme += f"## Dataset Info\n\n"
        readme += f"- **Sequence Length**: {ds['seq_len']}\n"
        readme += f"- **Variates**: {ds['num_variates']}\n"
        readme += f"- **Classes**: {ds['num_classes']}\n"
        readme += f"- **Train Samples**: {ds['num_train']}\n"
        readme += f"- **Test Samples**: {ds['num_test']}\n"
        readme += f"- **Trigger Epochs**: {get_trigger_epochs(ds['num_classes'])} (15 × {ds['num_classes']})\n\n"
        
        readme += f"## Experiments\n\n"
        readme += f"**Total Experiments**: {dataset_total}\n\n"
        
        readme += f"### By Method\n\n"
        for method, count in method_counts.items():
            readme += f"- **{method}**: {count} experiments → `{method}.sh`\n"
        
        readme += f"\n## CNN Trigger Models\n\n"
        readme += f"- **ccnn**: Conditional CNN trigger network\n"
        readme += f"- **ccnn_cae**: Conditional CNN with autoencoder\n\n"
        readme += f"Note: CNN models don't require transformer architecture parameters\n"
        readme += f"(d_model_bd, d_ff_bd, e_layers_bd, n_heads_bd, patch_len, stride)\n\n"
        
        readme += f"## How to Run\n\n"
        readme += f"### Run All Methods\n"
        readme += f"```bash\n"
        readme += f"bash run_all_methods.sh\n"
        readme += f"```\n\n"
        
        readme += f"### Run Specific Method\n"
        readme += f"```bash\n"
        readme += f"bash marksman.sh  # or any other method\n"
        readme += f"```\n\n"
        
        readme += f"### Run Subset\n"
        readme += f"```bash\n"
        readme += f"# First 50 experiments only\n"
        readme += f"head -100 marksman.sh > marksman_subset.sh\n"
        readme += f"bash marksman_subset.sh\n"
        readme += f"```\n"
        
        readme_file = os.path.join(dataset_dir, 'README.md')
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme)
        
        dataset_summaries.append({
            'name': dataset_name,
            'total': dataset_total,
            'methods': method_counts
        })
        
        total_experiments_all += dataset_total
        print(f"  Total for {dataset_name}: {dataset_total} experiments")
    
    # Generate overall summary
    summary = "# CNN Trigger Network Parameter Exploration\n\n"
    summary += "## Organization: Per-Dataset Scripts\n\n"
    summary += "Scripts are organized by dataset for focused exploration.\n"
    summary += "Each dataset has its own directory with method-specific scripts.\n\n"
    
    summary += f"## Overview\n\n"
    summary += f"**Total Experiments (All Datasets)**: {total_experiments_all}\n"
    summary += f"**Datasets**: {len(datasets)}\n"
    summary += f"**Trigger Models**: {', '.join(TRIGGER_MODELS)}\n"
    summary += f"**Methods**: {', '.join(TRAINING_METHODS)}\n\n"
    
    summary += "## Key Differences from Transformer Models\n\n"
    summary += "CNN-based triggers (ccnn, ccnn_cae) have simpler architecture:\n"
    summary += "- **No transformer parameters**: d_model_bd, d_ff_bd, e_layers_bd, n_heads_bd\n"
    summary += "- **No patch parameters**: patch_len, stride\n"
    summary += "- **Fewer total experiments** per dataset due to simpler architecture\n"
    summary += "- **Same optimizer exploration**: 10 optimizer configurations\n"
    summary += "- **Same method exploration**: All training methods supported\n\n"
    
    summary += "## Datasets\n\n"
    for ds_sum in dataset_summaries:
        summary += f"### {ds_sum['name']}\n"
        summary += f"**Total**: {ds_sum['total']} experiments\n"
        summary += f"**Location**: `cnn_per_dataset/{ds_sum['name']}/`\n\n"
        summary += "Methods:\n"
        for method, count in ds_sum['methods'].items():
            summary += f"- {method}: {count} experiments\n"
        summary += "\n"
    
    summary += "## Parameter Space\n\n"
    
    summary += "### Optimizer Configurations (10)\n\n"
    for name, config in OPTIMIZER_CONFIGS.items():
        summary += f"**{name}**: {config['trigger_opt']}/{config['surrogate_opt']}\n"
    summary += "\n"
    
    summary += "### Training Dynamics\n\n"
    summary += f"- **Warmup**: {WARMUP_EPOCHS}\n"
    summary += f"- **Trigger Epochs**: Dynamic (15 × num_classes per dataset)\n"
    summary += f"- **Batch Sizes**: {BATCH_SIZES}\n\n"
    
    summary += "### Poisoning Strategies\n\n"
    for config in POISONING_CONFIGS:
        summary += f"- **{config['name']}**\n"
    summary += "\n"
    
    summary += "## How to Use\n\n"
    summary += "### Run All Experiments for One Dataset\n"
    summary += "```bash\n"
    summary += "cd cnn_per_dataset/BasicMotions\n"
    summary += "bash run_all_methods.sh\n"
    summary += "```\n\n"
    
    summary += "### Run Specific Method on One Dataset\n"
    summary += "```bash\n"
    summary += "cd cnn_per_dataset/BasicMotions\n"
    summary += "bash marksman.sh\n"
    summary += "```\n\n"
    
    summary += "### Run Same Method Across All Datasets\n"
    summary += "```bash\n"
    summary += "for dataset in cnn_per_dataset/*/; do\n"
    summary += "  cd \"$dataset\"\n"
    summary += "  bash marksman.sh\n"
    summary += "  cd ../..\n"
    summary += "done\n"
    summary += "```\n\n"
    
    summary += "## Comparison with Transformer Models\n\n"
    summary += "| Aspect | Transformer (citst/cpatchtst) | CNN (ccnn/ccnn_cae) |\n"
    summary += "|--------|-------------------------------|---------------------|\n"
    summary += "| Architecture Params | 6 configs × 4 params | None (simpler) |\n"
    summary += "| Patch Params | patch_len, stride | None |\n"
    summary += "| Experiments per Dataset | ~9,600 | ~1,600 |\n"
    summary += "| Training Speed | Slower | Faster |\n"
    summary += "| Capacity | Higher | Lower |\n"
    
    # Save overall summary
    summary_file = os.path.join(base_output_dir, 'README.md')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\n{'='*70}")
    print(f"✓ CNN PER-DATASET SCRIPTS GENERATED")
    print(f"{'='*70}")
    print(f"Total Experiments: {total_experiments_all}")
    print(f"Datasets: {len(datasets)}")
    print(f"Trigger Models: {len(TRIGGER_MODELS)}")
    print(f"Methods: {len(TRAINING_METHODS)}")
    print(f"Optimizer Configs: {len(OPTIMIZER_CONFIGS)}")
    print(f"\nOutput Directory: scripts/cnn_per_dataset/")
    print(f"Summary: scripts/cnn_per_dataset/README.md")
    print(f"{'='*70}\n")
    
    print(f"Trigger epochs are dynamically calculated: 15 × num_classes per dataset")
    print(f"Each dataset has its own directory with {len(TRAINING_METHODS)} method scripts")
    print(f"CNN models have simpler architecture (no transformer params)")
    print(f"Run experiments on one dataset at a time for focused exploration!")

if __name__ == '__main__':
    main()
