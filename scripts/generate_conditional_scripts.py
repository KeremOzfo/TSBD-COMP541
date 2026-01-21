"""
Script to generate conditional backdoor training scripts for all datasets.
Tests conditional models (cpatchtst, citst, ccnn, ccnn_cae) with all2all attack.
Tries all training methods with different hyperparameter configurations.
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

# Conditional trigger models
CONDITIONAL_MODELS = ['cpatchtst', 'citst', 'ccnn', 'ccnn_cae']

# All training methods
TRAINING_METHODS = ['basic', 'inputaware', 'pureinputaware', 'marksman', 'frequency', 'ultimate']

# Optimizer presets based on literature review of backdoor attacks
# Reference: WaNet, LIRA, Input-Aware, Blind Backdoor, ISSBA papers
OPTIMIZER_PRESETS = {
    'standard': {
        'name': 'Standard',
        'trigger_opt': 'adam',
        'trigger_lr': 1e-3,
        'trigger_weight_decay': 0,
        'surrogate_opt': 'adam',
        'surrogate_lr': 1e-3,
        'surrogate_weight_decay': 0,
        'description': 'Standard configuration used in most papers (Adam for both)'
    },
    'regularized': {
        'name': 'Regularized',
        'trigger_opt': 'adam',
        'trigger_lr': 1e-4,
        'trigger_weight_decay': 1e-5,
        'surrogate_opt': 'adamw',
        'surrogate_lr': 5e-4,
        'surrogate_weight_decay': 1e-2,
        'description': 'LIRA-style with regularization to prevent overfitting'
    },
    'aggressive': {
        'name': 'Aggressive',
        'trigger_opt': 'adamw',
        'trigger_lr': 1e-3,
        'trigger_weight_decay': 1e-4,
        'surrogate_opt': 'sgd',
        'surrogate_lr': 1e-2,
        'surrogate_weight_decay': 5e-3,
        'description': 'Input-Aware style with aggressive learning and momentum'
    }
}

# Hyperparameter configurations per method (max 2 variants)
METHOD_HYPERPARAMS = {
    'basic': [
        {'clip_ratio': 0.1, 'name': 'default'}
    ],
    'inputaware': [
        {'p_attack': 0.5, 'p_cross': 0.1, 'lambda_cross': 1.0, 'clip_ratio': 0.1, 'name': 'default'},
        {'p_attack': 0.7, 'p_cross': 0.15, 'lambda_cross': 2.0, 'clip_ratio': 0.1, 'name': 'aggressive'}
    ],
    'pureinputaware': [
        {'p_attack': 0.5, 'clip_ratio': 0.1, 'name': 'default'},
        {'p_attack': 0.7, 'clip_ratio': 0.1, 'name': 'high_attack'}
    ],
    'marksman': [
        {'clip_ratio': 0.1, 'marksman_alpha': 0.5, 'marksman_beta': 0.0, 'marksman_update_T': 1, 'name': 'default'},
        {'clip_ratio': 0.1, 'marksman_alpha': 0.7, 'marksman_beta': 0.001, 'marksman_update_T': 2, 'name': 'high_alpha'},
        {'clip_ratio': 0.1, 'marksman_alpha': 0.5, 'marksman_beta': 1, 'marksman_update_T': 1, 'name': 'high_beta'},
        {'clip_ratio': 0.1, 'marksman_alpha': 0.5, 'marksman_beta': 0.1, 'marksman_update_T': 2, 'name': 'high_update'}
    ],
    'diversity': [
        {'clip_ratio': 0.1, 'div_reg': 1.0, 'poisoning_ratio_train': 0.1, 'name': 'default'},
        {'clip_ratio': 0.1, 'div_reg': 2.0, 'poisoning_ratio_train': 0.2, 'name': 'high_div'}
    ],
    'frequency': [
        {'clip_ratio': 0.1, 'lambda_freq': 1.0, 'freq_lambda': 0.05, 'name': 'default'},
        {'clip_ratio': 0.1, 'lambda_freq': 2.0, 'freq_lambda': 0.1, 'name': 'high_freq'}
    ],
    'ultimate': [
        {'clip_ratio': 0.1, 'lambda_freq': 1.0, 'lambda_div': 1.0, 'lambda_reg': 1e-3, 'lambda_cross': 1.0, 'p_attack': 0.5, 'p_cross': 0.1, 'name': 'balanced'},
        {'clip_ratio': 0.1, 'lambda_freq': 2.0, 'lambda_div': 2.0, 'lambda_reg': 1e-2, 'lambda_cross': 2.0, 'p_attack': 0.7, 'p_cross': 0.15, 'name': 'aggressive'}
    ]
}

def get_batch_size(num_train, seq_len):
    """Determine batch size based on dataset characteristics."""
    if num_train < 100:
        return 8
    elif num_train < 500:
        return 16
    elif seq_len > 1000:
        return 8
    elif seq_len > 500:
        return 16
    else:
        return 32

def get_trigger_config(seq_len, num_variates):
    """Determine trigger network configuration."""
    if seq_len <= 100:
        config = {
            'patch_len': max(4, seq_len // 8),
            'stride': max(2, seq_len // 16),
            'd_model_bd': 16,
            'd_ff_bd': 32,
            'e_layers_bd': 1,
            'n_heads_bd': 2
        }
    elif seq_len <= 500:
        config = {
            'patch_len': 16,
            'stride': 8,
            'd_model_bd': 32,
            'd_ff_bd': 64,
            'e_layers_bd': 1,
            'n_heads_bd': 4
        }
    else:
        config = {
            'patch_len': 32,
            'stride': 16,
            'd_model_bd': 64,
            'd_ff_bd': 128,
            'e_layers_bd': 2,
            'n_heads_bd': 4
        }
    
    # Scale for high-dimensional data
    if num_variates > 50:
        config['d_model_bd'] = min(config['d_model_bd'] * 2, 128)
        config['d_ff_bd'] = min(config['d_ff_bd'] * 2, 256)
        config['n_heads_bd'] = 8
    elif num_variates > 10:
        config['d_model_bd'] = min(int(config['d_model_bd'] * 1.5), 96)
        config['d_ff_bd'] = min(int(config['d_ff_bd'] * 1.5), 192)
    
    return config

def get_training_epochs(num_train, seq_len, num_classes):
    """
    Determine training epochs based on dataset complexity.
    
    Factors:
    - Dataset size: Small datasets need more epochs
    - Sequence length: Longer sequences need more epochs
    - Number of classes: More classes = harder conditional generation
    """
    SMALL_DATASET_THRESHOLD = 200
    
    # Base epochs based on dataset size
    if num_train < SMALL_DATASET_THRESHOLD:
        train_epochs = 30
        bd_epochs = 30
        trigger_epochs = 60
    elif num_train < 1000:
        train_epochs = 30
        bd_epochs = 30
        trigger_epochs = 60
    else:
        train_epochs = 30
        bd_epochs = 30
        trigger_epochs = 60
    
    # Adjust for long sequences (more complex patterns)
    if seq_len > 1000:
        train_epochs = int(train_epochs * 1.5)
        bd_epochs = int(bd_epochs * 1.5)
        trigger_epochs = int(trigger_epochs * 1.2)
    
    # Scale trigger epochs for multi-class conditional generation
    # More classes = harder to learn class-specific triggers
    if num_classes > 20:
        trigger_epochs = int(trigger_epochs * 2.0)
    elif num_classes > 10:
        trigger_epochs = int(trigger_epochs * 1.5)
    elif num_classes > 5:
        trigger_epochs = int(trigger_epochs * 1.2)
    
    return train_epochs, bd_epochs, trigger_epochs

def generate_conditional_script(dataset, tmodel, method, hyperparam_config, optimizer_preset):
    """Generate conditional backdoor training script."""
    name = dataset['name']
    seq_len = dataset['seq_len']
    num_train = dataset['num_train']
    num_variates = dataset['num_variates']
    num_classes = dataset['num_classes']
    
    batch_size = get_batch_size(num_train, seq_len)
    train_epochs, bd_epochs, trigger_epochs = get_training_epochs(num_train, seq_len, num_classes)
    trigger_config = get_trigger_config(seq_len, num_variates)
    
    # Base command
    cmd = f"python -u main.py --train_mode backdoor --method {method} --Tmodel {tmodel} --bd_type all2all"
    cmd += f" --root_path ./dataset/{name}"
    cmd += f" --seq_len {seq_len}"
    cmd += f" --batch_size {batch_size}"
    cmd += f" --trigger_epochs {trigger_epochs}"
    cmd += f" --bd_train_epochs {bd_epochs}"
    cmd += f" --poisoning_ratio 0.1"
    cmd += f" --target_label 0"
    
    # Trigger model config
    cmd += f" --d_model_bd {trigger_config['d_model_bd']}"
    cmd += f" --d_ff_bd {trigger_config['d_ff_bd']}"
    cmd += f" --e_layers_bd {trigger_config['e_layers_bd']}"
    cmd += f" --n_heads_bd {trigger_config['n_heads_bd']}"
    cmd += f" --patch_len {trigger_config['patch_len']}"
    cmd += f" --stride {trigger_config['stride']}"
    
    # Apply optimizer preset
    cmd += f" --trigger_opt {optimizer_preset['trigger_opt']}"
    cmd += f" --trigger_lr {optimizer_preset['trigger_lr']}"
    cmd += f" --trigger_weight_decay {optimizer_preset['trigger_weight_decay']}"
    cmd += f" --surrogate_opt {optimizer_preset['surrogate_opt']}"
    cmd += f" --surrogate_lr {optimizer_preset['surrogate_lr']}"
    cmd += f" --surrogate_weight_decay {optimizer_preset['surrogate_weight_decay']}"
    
    # Add method-specific hyperparameters
    for key, value in hyperparam_config.items():
        if key != 'name':
            cmd += f" --{key} {value}"
    
    return cmd + "\n"

def main():
    # Create output directory
    output_dir = os.path.join(script_dir, 'conditional_all2all')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate master script for all experiments
    master_script = "#!/bin/bash\n"
    master_script += "# Conditional Backdoor Experiments: all2all attack\n"
    master_script += "# Models: cpatchtst, citst, ccnn, ccnn_cae\n"
    master_script += "# Methods: basic, inputaware, pureinputaware, marksman, diversity, frequency, ultimate\n"
    master_script += "# Datasets: from dataset_info_subset.csv\n\n"
    
    total_experiments = 0
    
    # Generate scripts by trigger model
    for tmodel in CONDITIONAL_MODELS:
        model_script = f"#!/bin/bash\n# Conditional Model: {tmodel} (all2all)\n\n"
        
        for method in TRAINING_METHODS:
            method_script = f"#!/bin/bash\n# Method: {method} with {tmodel} (all2all)\n\n"
            
            for hyperparam_config in METHOD_HYPERPARAMS[method]:
                config_name = hyperparam_config['name']
                
                for preset_key, optimizer_preset in OPTIMIZER_PRESETS.items():
                    preset_name = optimizer_preset['name']
                    method_script += f"# ========== {method.upper()} - {config_name.upper()} - {preset_name.upper()} ==========\n"
                    model_script += f"# ========== {method.upper()} - {config_name.upper()} - {preset_name.upper()} ==========\n"
                    
                    for ds in datasets:
                        comment = f"# {ds['name']}: seq={ds['seq_len']}, vars={ds['num_variates']}, classes={ds['num_classes']}, train={ds['num_train']}\n"
                        method_script += comment
                        model_script += comment
                        
                        cmd = generate_conditional_script(ds, tmodel, method, hyperparam_config, optimizer_preset)
                        method_script += cmd + "\n"
                        model_script += cmd + "\n"
                        master_script += f"# {tmodel} - {method} - {config_name} - {preset_name} - {ds['name']}\n"
                        master_script += cmd + "\n"
                        
                        total_experiments += 1
                    
                    method_script += "\n"
                    model_script += "\n"
            
            # Save by method
            method_file = os.path.join(output_dir, f'method_{method}_{tmodel}.sh')
            with open(method_file, 'w', newline='\n') as f:
                f.write(method_script)
        
        # Save by model
        model_file = os.path.join(output_dir, f'model_{tmodel}_all_methods.sh')
        with open(model_file, 'w', newline='\n') as f:
            f.write(model_script)
    
    # Generate scripts by dataset
    for ds in datasets:
        ds_script = f"#!/bin/bash\n# Dataset: {ds['name']} (all2all conditional backdoor)\n"
        ds_script += f"# Seq={ds['seq_len']}, Vars={ds['num_variates']}, Classes={ds['num_classes']}\n\n"
        
        for tmodel in CONDITIONAL_MODELS:
            ds_script += f"# ========== TRIGGER MODEL: {tmodel.upper()} ==========\n"
            
            for method in TRAINING_METHODS:
                ds_script += f"\n# Method: {method}\n"
                
                for hyperparam_config in METHOD_HYPERPARAMS[method]:
                    config_name = hyperparam_config['name']
                    ds_script += f"# Config: {config_name}\n"
                    
                    for preset_key, optimizer_preset in OPTIMIZER_PRESETS.items():
                        preset_name = optimizer_preset['name']
                        ds_script += f"# Optimizer Preset: {preset_name}\n"
                        
                        cmd = generate_conditional_script(ds, tmodel, method, hyperparam_config, optimizer_preset)
                        ds_script += cmd + "\n"
            
            ds_script += "\n"
        
        # Save by dataset
        ds_file = os.path.join(output_dir, f'dataset_{ds["name"]}.sh')
        with open(ds_file, 'w', newline='\n') as f:
            f.write(ds_script)
    
    # Save master script
    master_file = os.path.join(output_dir, 'master_all_experiments.sh')
    with open(master_file, 'w', newline='\n') as f:
        f.write(master_script)
    
    # Generate summary
    summary = "# Conditional Backdoor Experiments Summary\n\n"
    summary += "## Configuration\n"
    summary += f"- **Attack Type**: all2all (class-conditional backdoors)\n"
    summary += f"- **Conditional Models**: {', '.join(CONDITIONAL_MODELS)}\n"
    summary += f"- **Training Methods**: {', '.join(TRAINING_METHODS)}\n"
    summary += f"- **Optimizer Presets**: {', '.join([p['name'] for p in OPTIMIZER_PRESETS.values()])}\n"
    summary += f"- **Datasets**: {len(datasets)} datasets\n"
    summary += f"- **Total Experiments**: {total_experiments}\n\n"
    
    summary += "## Optimizer Presets (Literature-Based)\n\n"
    for preset_key, preset in OPTIMIZER_PRESETS.items():
        summary += f"### {preset['name']} ({preset_key})\n"
        summary += f"**Description**: {preset['description']}\n"
        summary += f"- **Trigger Optimizer**: {preset['trigger_opt']} (lr={preset['trigger_lr']}, wd={preset['trigger_weight_decay']})\n"
        summary += f"- **Surrogate Optimizer**: {preset['surrogate_opt']} (lr={preset['surrogate_lr']}, wd={preset['surrogate_weight_decay']})\n"
        summary += "\n"
    
    summary += "## Hyperparameter Configurations per Method\n\n"
    for method in TRAINING_METHODS:
        summary += f"### {method}\n"
        for config in METHOD_HYPERPARAMS[method]:
            summary += f"- **{config['name']}**: "
            params = [f"{k}={v}" for k, v in config.items() if k != 'name']
            summary += ", ".join(params) + "\n"
        summary += "\n"
    
    summary += "## Dataset Overview\n\n"
    summary += "| Dataset | Seq Len | Variates | Classes | Train | Test |\n"
    summary += "|---------|---------|----------|---------|-------|------|\n"
    for ds in datasets:
        summary += f"| {ds['name']} | {ds['seq_len']} | {ds['num_variates']} | {ds['num_classes']} | {ds['num_train']} | {ds['num_test']} |\n"
    
    summary += "\n## Script Organization\n\n"
    summary += "### By Trigger Model\n"
    for tmodel in CONDITIONAL_MODELS:
        summary += f"- `model_{tmodel}_all_methods.sh`: All methods and optimizer presets for {tmodel}\n"
    
    summary += "\n### By Training Method\n"
    for method in TRAINING_METHODS:
        for tmodel in CONDITIONAL_MODELS:
            summary += f"- `method_{method}_{tmodel}.sh`: {method} with all optimizer presets using {tmodel}\n"
    
    summary += "\n### By Dataset\n"
    for ds in datasets:
        summary += f"- `dataset_{ds['name']}.sh`: All experiments for {ds['name']}\n"
    
    summary += "\n### Master Script\n"
    summary += f"- `master_all_experiments.sh`: All {total_experiments} experiments\n"
    
    # Save summary
    summary_file = os.path.join(output_dir, 'CONDITIONAL_SUMMARY.md')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"✓ Generated conditional backdoor scripts:")
    print(f"  - Total experiments: {total_experiments}")
    print(f"  - Models: {len(CONDITIONAL_MODELS)}")
    print(f"  - Methods: {len(TRAINING_METHODS)}")
    print(f"  - Datasets: {len(datasets)}")
    print(f"  - Output: scripts/conditional_all2all/")
    print(f"  - Summary: scripts/conditional_all2all/CONDITIONAL_SUMMARY.md")

if __name__ == '__main__':
    main()
