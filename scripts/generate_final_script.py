"""
Rigorous Trigger Network Parameter Exploration Script Generator

This script generates comprehensive training scripts to explore:
- Trigger model architectures (citst, cpatchtst)
- Network dimensions (d_model_bd, d_ff_bd, e_layers_bd, n_heads_bd)
- Training methods (basic, vanilla, marksman, diversity, ultimate, frequency, inputaware, pureinputaware)
- Optimizer configurations (type, lr, weight decay, momentum)
- Training dynamics (warmup, epochs, batch size)
- Regularization (L2 penalty, gradient clipping)
- Poisoning strategies (ratio, silent poisoning)

Goal: Identify optimal configurations for each method and understand what works.
"""
import csv
import os
import itertools

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

# Focus on citst and cpatchtst
#TRIGGER_MODELS = ['citst', 'cpatchtst','ccnn','ccnn_cae']

TRIGGER_MODELS = ['citst', 'cpatchtst', 'ctimesnet']
# Main classifier models to explore
MAIN_MODELS = ['TimesNet','TCN','lstm','nonstationary_transformer','mlp','resnet','iTransformer']

# All methods except defeat
TRAINING_METHODS = ['vanilla', 'marksman', 'pureinputaware']


# ==================== TOP 3 OPTIMIZER CONFIGURATIONS (Based on Analysis) ====================
# Empirical results show these are the best performing configurations
# Keeping only these 3 to reduce search space while maintaining performance
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
    'basic': [
        {'name': 'default', 'params': {}},
    ],
    'vanilla': [
        {'name': 'default', 'params': {}},
    ],
    'marksman': [
        {'name': 'balanced', 'params': {'marksman_alpha': 0.5, 'marksman_beta': 0.0, 'marksman_update_T': 1}},
        {'name': 'with_penalty', 'params': {'marksman_alpha': 0.5, 'marksman_beta': 0.001, 'marksman_update_T': 1}},
        {'name': 'slow_update', 'params': {'marksman_alpha': 0.5, 'marksman_beta': 0.0, 'marksman_update_T': 3}},
        {'name': 'slow_update_with_penalty', 'params': {'marksman_alpha': 0.5, 'marksman_beta': 0.001, 'marksman_update_T': 3}},
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
    ],
    'pureinputaware': [
        {'name': 'balanced', 'params': {'p_attack': 0.5, 'lambda_div': 1.0, 'lambda_cross': 1.0}},    
        ],
    'ultimate': [
        {'name': 'balanced', 'params': {'lambda_freq': 1.0, 'lambda_div': 1.0, 'lambda_reg': 1e-3, 'lambda_cross': 1.0, 'p_attack': 0.5, 'p_cross': 0.1}},
        {'name': 'freq_focused', 'params': {'lambda_freq': 3.0, 'lambda_div': 0.5, 'lambda_reg': 1e-3, 'lambda_cross': 0.5, 'p_attack': 0.5, 'p_cross': 0.1}},
        {'name': 'div_focused', 'params': {'lambda_freq': 0.5, 'lambda_div': 3.0, 'lambda_reg': 1e-3, 'lambda_cross': 0.5, 'p_attack': 0.5, 'p_cross': 0.1}},
        {'name': 'cross_focused', 'params': {'lambda_freq': 0.5, 'lambda_div': 0.5, 'lambda_reg': 1e-3, 'lambda_cross': 3.0, 'p_attack': 0.5, 'p_cross': 0.2}},
        {'name': 'high_reg', 'params': {'lambda_freq': 1.0, 'lambda_div': 1.0, 'lambda_reg': 1e-2, 'lambda_cross': 1.0, 'p_attack': 0.5, 'p_cross': 0.1}},
        {'name': 'aggressive', 'params': {'lambda_freq': 2.0, 'lambda_div': 2.0, 'lambda_reg': 1e-3, 'lambda_cross': 2.0, 'p_attack': 0.7, 'p_cross': 0.15}},
    ]
}

# ==================== TRAINING DYNAMICS ====================
WARMUP_EPOCHS = [0, 5]  # Different warmup strategies

# ==================== GRADIENT CLIPPING CONFIGURATIONS ====================
GRAD_CLIP_CONFIGS = [
    {'trigger_grad_clip': 0.0, 'surrogate_grad_clip': 0.0, 'name': 'no_clip'},
    {'trigger_grad_clip': 5.0, 'surrogate_grad_clip': 5.0, 'name': 'clip_5'},
    {'trigger_grad_clip': 10.0, 'surrogate_grad_clip': 10.0, 'name': 'clip_10'},
]

# ==================== POISONING STRATEGIES ====================
POISONING_CONFIGS = [
    {'poisoning_ratio': 0.1, 'use_silent_poisoning': False, 'name': 'normal_10pct'},
    #{'poisoning_ratio': 0.2, 'use_silent_poisoning': False, 'name': 'normal_20pct'},
    #{'poisoning_ratio': 0.1, 'use_silent_poisoning': True, 'lambda_ratio': 0.2, 'name': 'silent_10pct_20clean'},
    #{'poisoning_ratio': 0.1, 'use_silent_poisoning': True, 'lambda_ratio': 0.5, 'name': 'silent_10pct_50clean'},
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
    return min(15 * num_classes, 200)

def get_patch_config(seq_len):
    """Get patch length and stride based on sequence length."""
    if seq_len <= 100:
        return max(4, seq_len // 8), max(2, seq_len // 16)
    elif seq_len <= 500:
        return 16, 8
    else:
        return 32, 16

def generate_script_command(dataset, tmodel, method, opt_config, method_params, 
                            warmup, trigger_epochs, poison_config, grad_clip_config):
    """Generate a single training command with all parameters."""
    name = dataset['name']
    seq_len = dataset['seq_len']
    
    patch_len, stride = get_patch_config(seq_len)
    
    # Base command
    cmd = f"python -u main.py --train_mode backdoor --method {method} --Tmodel {tmodel} --bd_type all2all"
    cmd += f" --root_path ./dataset/{name}"
    cmd += f" --seq_len {seq_len}"
    cmd += f" --trigger_epochs {trigger_epochs}"
    cmd += f" --bd_train_epochs 40"
    cmd += f" --warmup_epochs {warmup}"
    cmd += f" --target_label 0"
    
    # Enable auto architecture and batch size selection
    cmd += " --auto_bd_arch True --auto_batch_size True"
    
    # Patch parameters (still needed for patching models)
    cmd += f" --patch_len {patch_len}"
    cmd += f" --stride {stride}"
    
    # Optimizer parameters
    cmd += f" --trigger_opt {opt_config['trigger_opt']}"
    cmd += f" --trigger_lr {opt_config['trigger_lr']}"
    cmd += f" --trigger_weight_decay {opt_config['trigger_weight_decay']}"
    cmd += f" --surrogate_opt {opt_config['surrogate_opt']}"
    cmd += f" --surrogate_lr {opt_config['surrogate_lr']}"
    cmd += f" --surrogate_weight_decay {opt_config['surrogate_weight_decay']}"
    cmd += f" --surrogate_L2_penalty {opt_config['surrogate_L2_penalty']}"
    
    # Gradient clipping parameters
    cmd += f" --trigger_grad_clip {grad_clip_config['trigger_grad_clip']}"
    cmd += f" --surrogate_grad_clip {grad_clip_config['surrogate_grad_clip']}"
    
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
    """Generate comprehensive rigorous exploration scripts - ONE DATASET AT A TIME."""
    base_output_dir = os.path.join(script_dir, 'final_rigorous_scripts')
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
        method_scripts = {}
        
        # Generate individual method scripts
        for method in TRAINING_METHODS:
            method_script = f"#!/bin/bash\n"
            method_script += f"# Dataset: {dataset_name}\n"
            method_script += f"# Method: {method}\n"
            method_script += f"# Seq Len: {ds['seq_len']}, Variates: {ds['num_variates']}, Classes: {ds['num_classes']}\n\n"
            
            method_count = 0
            
            for tmodel in TRIGGER_MODELS:
                method_script += f"\n# ========== TRIGGER MODEL: {tmodel} ==========\n\n"
                
                # Calculate trigger epochs based on number of classes
                trigger_epochs = get_trigger_epochs(ds['num_classes'])
                
                # Iterate through all configurations
                for opt_name, opt_config in OPTIMIZER_CONFIGS.items():
                    for method_config in METHOD_HYPERPARAMS[method]:
                        for warmup in WARMUP_EPOCHS:
                            for grad_clip_config in GRAD_CLIP_CONFIGS:
                                for poison_config in POISONING_CONFIGS:
                                    
                                    # Generate comment
                                    comment = f"# {tmodel} | opt={opt_name} | "
                                    comment += f"cfg={method_config['name']} | warmup={warmup} | "
                                    comment += f"grad_clip={grad_clip_config['name']} | "
                                    comment += f"epochs={trigger_epochs} | poison={poison_config['name']}\n"
                                    
                                    method_script += comment
                                    
                                    # Generate command
                                    cmd = generate_script_command(
                                        ds, tmodel, method, opt_config,
                                        method_config['params'], warmup, trigger_epochs,
                                        poison_config, grad_clip_config
                                    )
                                
                                    method_script += cmd + "\n"
                                    
                                    method_count += 1
                                    dataset_total += 1
            
            # Save individual method script
            method_file = os.path.join(dataset_dir, f'{method}.sh')
            with open(method_file, 'w', newline='\n') as f:
                f.write(method_script)
            
            method_scripts[method] = method_script
            method_counts[method] = method_count
            print(f"  ✓ {method}: {method_count} experiments")
        
        # Concatenate all method scripts into one combined script
        combined_script = f"#!/bin/bash\n"
        combined_script += f"# RIGOROUS PARAMETER EXPLORATION - ALL METHODS\n"
        combined_script += f"# Dataset: {dataset_name}\n"
        combined_script += f"# Total Experiments: {dataset_total}\n"
        combined_script += f"# Seq Len: {ds['seq_len']}, Variates: {ds['num_variates']}, Classes: {ds['num_classes']}\n\n"
        
        for method in TRAINING_METHODS:
            combined_script += f"\n{'#'*70}\n"
            combined_script += f"# METHOD: {method.upper()}\n"
            combined_script += f"{'#'*70}\n\n"
            # Add the method script content (skip the shebang and header)
            lines = method_scripts[method].split('\n')
            # Skip first 4 lines (shebang + 3 comment lines)
            combined_script += '\n'.join(lines[4:]) + '\n'
        
        # Save combined script as both files (run_all_methods.sh and {dataset_name}_all_methods.sh)
        # run_all_methods.sh contains all python commands concatenated
        master_file = os.path.join(dataset_dir, 'run_all_methods.sh')
        with open(master_file, 'w', newline='\n') as f:
            f.write(combined_script)
        
        # Also save as {dataset_name}_all_methods.sh for consistency
        combined_file = os.path.join(dataset_dir, f'{dataset_name}_all_methods.sh')
        with open(combined_file, 'w', newline='\n') as f:
            f.write(combined_script)
        
        # Generate dataset-specific README
        readme = f"# Rigorous Exploration: {dataset_name}\n\n"
        readme += f"## Dataset Info\n\n"
        readme += f"- **Sequence Length**: {ds['seq_len']}\n"
        readme += f"- **Master script (calls all methods)**: `run_all_methods.sh`\n"
        readme += f"- **Combined script (concatenated)**: `{dataset_name}_all_methods.sh`\n\n"
        
        readme += f"## How to Run\n\n"
        readme += f"### Run All Experiments (Sequential by Method)\n"
        readme += f"```bash\n"
        readme += f"bash run_all_methods.sh\n"
        readme += f"```\n\n"
        
        readme += f"### Run All Experiments (Concatenated\n\n"
        
        readme += f"## Experiments\n\n"
        readme += f"**Total Experiments**: {dataset_total}\n\n"
        
        readme += f"### By Method\n\n"
        for method, count in method_counts.items():
            readme += f"- **{method}**: {count} experiments → `{method}.sh`\n"
        
        readme += f"\n## Available Scripts\n\n"
        readme += f"- **Individual method scripts**: `{', '.join([m + '.sh' for m in TRAINING_METHODS])}`\n"
        readme += f"- **All methods (concatenated)**: `run_all_methods.sh` or `{dataset_name}_all_methods.sh` (same file)\n\n"
        
        readme += f"## How to Run\n\n"
        readme += f"### Run All Experiments (All Methods)\n"
        readme += f"```bash\n"
        readme += f"bash run_all_methods.sh\n"
        readme += f"# OR\n"
        readme += f"bash {dataset_name}_all_methods.sh\n"
        readme += f"```\n\n"
        
        readme += f"### Run Specific Method\n"
        readme += f"```bash\n"
        readme += f"bash vanilla.sh      # or marksman.sh, pureinputaware.sh\n"
        readme += f"```\n\n"
        
        readme += f"### Run Subset\n"
        readme += f"```bash\n"
        readme += f"# First 100 experiments only\n"
        readme += f"head -200 {dataset_name}_all_methods.sh > {dataset_name}_subset.sh\n"
        readme += f"bash {dataset_name}_subset.sh\n"
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
    summary = "# Rigorous Trigger Network Parameter Exploration\n\n"
    summary += "## Organization: Per-Dataset Scripts\n\n"
    summary += "Scripts are organized by dataset for focused exploration.\n"
    summary += "Each dataset has its own directory with method-specific scripts.\n\n"
    
    summary += f"## Overview\n\n"
    summary += f"**Total Experiments (All Datasets)**: {total_experiments_all}\n"
    summary += f"**Datasets**: {len(datasets)}\n"
    summary += f"**Trigger Models**: {', '.join(TRIGGER_MODELS)}\n"
    summary += f"**Methods**: {', '.join(TRAINING_METHODS)}\n\n"
    
    summary += "## Datasets\n\n"
    for ds_sum in dataset_summaries:
        summary += f"### {ds_sum['name']}\n"
        summary += f"**Total**: {ds_sum['total']} experiments\n"
        summary += f"**Script**: `rigorous_per_dataset/{ds_sum['name']}/{ds_sum['name']}_all_methods.sh`\n\n"
        summary += "Methods:\n"
        for method, count in ds_sum['methods'].items():
            summary += f"- {method}: {count} experiments\n"
        summary += "\n"
    
    summary += "## Parameter Space\n\n"
    
    summary += "### Architecture & Batch Size\n\n"
    summary += "- **Auto Architecture Selection**: Enabled (based on dataset complexity)\n"
    summary += "- **Auto Batch Size Selection**: Enabled (based on dataset statistics)\n\n"
    
    summary += "### Optimizer Configurations\n\n"
    for name, config in OPTIMIZER_CONFIGS.items():
        summary += f"**{name}**: {config['trigger_opt']}/{config['surrogate_opt']}\n"
    summary += "\n"
    
    summary += "### Training Dynamics\n\n"
    summary += f"- **Warmup**: {WARMUP_EPOCHS}\n"
    summary += f"- **Trigger Epochs**: Dynamic (15 × num_classes per dataset)\n"
    summary += f"- **Batch Size**: Auto-selected based on dataset statistics\n"
    summary += f"- **Architecture**: Auto-selected based on dataset complexity\n\n"
    
    summary += "### Poisoning Strategies (4)\n\n"
    for config in POISONING_CONFIGS:
        summary += f"- **{config['name']}**\n"
    summary += "\n"
    
    summary += "## How to Use\n\n"
    summary += "### Run All Experiments for One Dataset\n"
    summary += "```bash\n"
    summary += "cd rigorous_per_dataset/BasicMotions\n"
    summary += "bash run_all_methods.sh\n"
    summary += "```\n\n"
    
    summary += "### Run All Experiments Across All Datasets\n"
    summary += "```bash\n"
    summary += "for dataset_dir in rigorous_per_dataset/*/; do\n"
    summary += "  dataset=$(basename \"$dataset_dir\")\n"
    summary += "  cd \"$dataset_dir\"\n"
    summary += "  bash \"${dataset}_all_methods.sh\"\n"
    summary += "  cd ../..\n"
    summary += "done\n"
    summary += "```\n\n"
    
    summary += "## Critical Parameters to Monitor\n\n"
    summary += "1. **Architecture Capacity** - d_model_bd, e_layers_bd\n"
    summary += "2. **Learning Rate** - trigger_lr, surrogate_lr\n"
    summary += "3. **Weight Decay / L2 Penalty**\n"
    summary += "4. **Warmup Epochs**\n"
    summary += "5. **Batch Size**\n"
    summary += "6. **Poisoning Strategy**\n"
    summary += "7. **Method-Specific Parameters**\n\n"
    
    summary += "## Missing Parameters to Consider\n\n"
    summary += "1. **Gradient Clipping**\n"
    summary += "2. **Learning Rate Scheduling**\n"
    summary += "3. **Trigger Model Dropout**\n"
    summary += "4. **Early Stopping**\n"
    summary += "5. **Data Augmentation**\n"
    
    # Save overall summary
    summary_file = os.path.join(base_output_dir, 'README.md')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\n{'='*70}")
    print(f"✓ PER-DATASET SCRIPTS GENERATED")
    print(f"{'='*70}")
    print(f"Total Experiments: {total_experiments_all}")
    print(f"Datasets: {len(datasets)}")
    print(f"Trigger Models: {len(TRIGGER_MODELS)}")
    print(f"Methods: {len(TRAINING_METHODS)}")
    print(f"Optimizer Configs: {len(OPTIMIZER_CONFIGS)}")
    print(f"\nOutput Directory: scripts/rigorous_per_dataset/")
    print(f"Summary: scripts/rigorous_per_dataset/README.md")
    print(f"{'='*70}\n")
    
    print(f"Architecture and batch size are auto-selected based on dataset characteristics")
    print(f"Trigger epochs are dynamically calculated: 15 × num_classes per dataset")
    print(f"Each dataset has individual method scripts + one combined script")
    print(f"Run experiments on one dataset at a time for focused exploration!")

if __name__ == '__main__':
    main()