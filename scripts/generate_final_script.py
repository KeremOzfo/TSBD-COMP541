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

TRIGGER_MODELS = ['citst', 'cpatchtst', 'ctimesnet','ccnn','ccnn_cae']
# Main classifier models to explore
MAIN_MODELS = ['TimesNet','TCN','lstm','mlp','resnet','FEDformer','Autoformer']

# All methods except defeat
TRAINING_METHODS = ['vanilla', 'marksman', 'pureinputaware']


# ==================== TOP 3 OPTIMIZER CONFIGURATIONS (Based on Analysis) ====================
# Empirical results show these are the best performing configurations
# Keeping only these 3 to reduce search space while maintaining performance
OPTIMIZER_CONFIGS_TOP3 = {
    # RANK 1: Best overall - Adam for trigger, SGD for surrogate (Mean CA: 0.8162)
    'adam_sgd_base': {
        'trigger_opt': 'adam',
        'trigger_lr': 1e-3,
        'trigger_weight_decay': 0.0,
        'surrogate_opt': 'sgd',
        'surrogate_lr': 1e-3,
        'surrogate_weight_decay': 5e-4,
        'surrogate_L2_penalty': 0.0,
        'description':'Base'
    },

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
        {'name': 'slow_update', 'params': {'marksman_alpha': 0.5, 'marksman_beta': 0.0, 'marksman_update_T': 3}},
    ],
    'pureinputaware': [
        {'name': 'balanced', 'params': {'p_attack': 0.5, 'lambda_div': 1.0, 'p_cross': 0.1}},    
        ],
}

# ==================== TRAINING DYNAMICS ====================
WARMUP_EPOCHS = [0]  # Different warmup strategies

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
    return min(10 * num_classes, 200)

def get_patch_config(seq_len):
    """Get patch length and stride based on sequence length."""
    if seq_len <= 100:
        return max(4, seq_len // 8), max(2, seq_len // 16)
    elif seq_len <= 500:
        return 16, 8
    else:
        return 32, 16

def generate_script_command(dataset, main_model, tmodel, method, opt_config, method_params, 
                            warmup, trigger_epochs, poison_config):
    """Generate a single training command with all parameters."""
    name = dataset['name']
    seq_len = dataset['seq_len']
    
    patch_len, stride = get_patch_config(seq_len)
    
    # Base command
    cmd = f"python -u main.py --train_mode backdoor --model {main_model} --method {method} --Tmodel {tmodel} --bd_type all2all"
    cmd += f" --root_path ./dataset/{name}"
    cmd += f" --seq_len {seq_len}"
    cmd += f" --trigger_epochs {trigger_epochs}"
    cmd += f" --trigger_patience 10"
    cmd += f" --bd_train_epochs 40"
    cmd += f" --warmup_epochs {warmup}"
    cmd += f" --target_label 0"
    
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
    
    # Conditional gradient clipping based on learning rate
    if opt_config['trigger_lr'] > 0.01:
        cmd += f" --trigger_grad_clip 10.0"
    else:
        cmd += f" --trigger_grad_clip 0.0"
    
    if opt_config['surrogate_lr'] > 0.01:
        cmd += f" --surrogate_grad_clip 10.0"
    else:
        cmd += f" --surrogate_grad_clip 0.0"
    
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
        
        # Generate single script for this dataset
        dataset_script = f"#!/bin/bash\n"
        dataset_script += f"# Dataset: {dataset_name}\n"
        dataset_script += f"# Seq Len: {ds['seq_len']}, Variates: {ds['num_variates']}, Classes: {ds['num_classes']}\n\n"
        
        for method in TRAINING_METHODS:
            dataset_script += f"\n{'#'*70}\n"
            dataset_script += f"# METHOD: {method.upper()}\n"
            dataset_script += f"{'#'*70}\n\n"
            
            method_count = 0
            
            for main_model in MAIN_MODELS:
                dataset_script += f"\n# ========== MAIN MODEL: {main_model} ==========\n\n"
                
                for tmodel in TRIGGER_MODELS:
                    # Skip citst for univariate datasets (citst requires multivariate data)
                    if tmodel == 'citst' and ds['num_variates'] == 1:
                        dataset_script += f"\n# ---- Trigger Model: {tmodel} (SKIPPED - univariate dataset) ----\n\n"
                        continue
                    
                    dataset_script += f"\n# ---- Trigger Model: {tmodel} ----\n\n"
                    
                    # Calculate trigger epochs based on number of classes
                    trigger_epochs = get_trigger_epochs(ds['num_classes'])
                    
                    # Iterate through all configurations
                    for opt_name, opt_config in OPTIMIZER_CONFIGS.items():
                        for method_config in METHOD_HYPERPARAMS[method]:
                            for warmup in WARMUP_EPOCHS:
                                for poison_config in POISONING_CONFIGS:
                                    
                                    # Determine grad clipping status for comment
                                    trigger_clip = "clip" if opt_config['trigger_lr'] > 0.01 else "no_clip"
                                    surrogate_clip = "clip" if opt_config['surrogate_lr'] > 0.01 else "no_clip"
                                    clip_status = f"{trigger_clip}/{surrogate_clip}"
                                    
                                    # Generate comment
                                    comment = f"# {method} | {main_model} | {tmodel} | opt={opt_name} | "
                                    comment += f"cfg={method_config['name']} | warmup={warmup} | "
                                    comment += f"grad_clip={clip_status} | "
                                    comment += f"epochs={trigger_epochs} | poison={poison_config['name']}\n"
                                    
                                    dataset_script += comment
                                    
                                    # Generate command
                                    cmd = generate_script_command(
                                        ds, main_model, tmodel, method, opt_config,
                                        method_config['params'], warmup, trigger_epochs,
                                        poison_config
                                    )
                                
                                    dataset_script += cmd + "\n"
                                    
                                    method_count += 1
                                    dataset_total += 1
            
            method_counts[method] = method_count
            print(f"  ✓ {method}: {method_count} experiments")
        
        # Save single dataset script
        dataset_file = os.path.join(dataset_dir, f'{dataset_name}.sh')
        with open(dataset_file, 'w', newline='\n') as f:
            f.write(dataset_script)
        
        # Generate dataset-specific README
        readme = f"# Rigorous Exploration: {dataset_name}\n\n"
        readme += f"## Dataset Info\n\n"
        readme += f"- **Sequence Length**: {ds['seq_len']}\n"
        readme += f"- **Number of Variates**: {ds['num_variates']}\n"
        readme += f"- **Number of Classes**: {ds['num_classes']}\n"
        readme += f"- **Training Samples**: {ds['num_train']}\n"
        readme += f"- **Test Samples**: {ds['num_test']}\n\n"
        
        readme += f"## Experiments\n\n"
        readme += f"**Total Experiments**: {dataset_total}\n\n"
        
        readme += f"### By Method\n\n"
        for method, count in method_counts.items():
            readme += f"- **{method}**: {count} experiments\n"
        
        readme += f"\n## Script\n\n"
        readme += f"- **Single dataset script**: `{dataset_name}.sh`\n\n"
        
        readme += f"## How to Run\n\n"
        readme += f"```bash\n"
        readme += f"bash {dataset_name}.sh\n"
        readme += f"```\n\n"
        
        readme += f"### Run Subset\n"
        readme += f"```bash\n"
        readme += f"# First 100 experiments only\n"
        readme += f"head -200 {dataset_name}.sh > {dataset_name}_subset.sh\n"
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
        summary += f"**Script**: `{ds_sum['name']}/{ds_sum['name']}.sh`\n\n"
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
    summary += "cd final_rigorous_scripts/BasicMotions\n"
    summary += "bash BasicMotions.sh\n"
    summary += "```\n\n"
    
    summary += "### Run All Experiments Across All Datasets\n"
    summary += "```bash\n"
    summary += "for dataset_dir in final_rigorous_scripts/*/; do\n"
    summary += "  dataset=$(basename \"$dataset_dir\")\n"
    summary += "  cd \"$dataset_dir\"\n"
    summary += "  bash \"${dataset}.sh\"\n"
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
    print(f"✓ DATASET SCRIPTS GENERATED")
    print(f"{'='*70}")
    print(f"Total Experiments: {total_experiments_all}")
    print(f"Datasets: {len(datasets)}")
    print(f"Trigger Models: {len(TRIGGER_MODELS)}")
    print(f"Methods: {len(TRAINING_METHODS)}")
    print(f"Optimizer Configs: {len(OPTIMIZER_CONFIGS)}")
    print(f"\nOutput Directory: scripts/final_rigorous_scripts/")
    print(f"Summary: scripts/final_rigorous_scripts/README.md")
    print(f"{'='*70}\n")
    
    print(f"Each dataset has ONE script: <dataset_name>.sh")
    print(f"Architecture and batch size are auto-selected")
    print(f"Trigger epochs: 10 × num_classes (max 200)")
    print(f"citst skipped for univariate datasets")

if __name__ == '__main__':
    main()