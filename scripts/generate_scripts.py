"""
Script to generate training scripts for all datasets based on dataset_info.csv.
Trigger networks perform short-term or long-term forecasting depending on sequence lengths.
"""
import csv
import os

# Read dataset info
datasets = []
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'dataset_info.csv')
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

# Define thresholds for categorization
SHORT_SEQ_THRESHOLD = 100      # <= 100: short sequence
MEDIUM_SEQ_THRESHOLD = 500     # <= 500: medium sequence
# > 500: long sequence

SMALL_DATASET_THRESHOLD = 200  # <= 200 train samples: small dataset

def get_batch_size(num_train, seq_len):
    """Determine batch size based on dataset and sequence characteristics."""
    if num_train < 100:
        return 8
    elif num_train < 500:
        return 16
    elif seq_len > 1000:
        return 8  # Large sequences need smaller batches
    elif seq_len > 500:
        return 16
    else:
        return 32

def get_model_config(seq_len, num_variates):
    """Get d_model and d_ff based on sequence complexity."""
    if seq_len <= SHORT_SEQ_THRESHOLD:
        d_model, d_ff = 32, 64
    elif seq_len <= MEDIUM_SEQ_THRESHOLD:
        d_model, d_ff = 64, 128
    else:
        d_model, d_ff = 128, 256
    
    # Adjust for high-dimensional data
    if num_variates > 50:
        d_model = min(d_model * 2, 256)
        d_ff = min(d_ff * 2, 512)
    
    return d_model, d_ff

def get_trigger_config(seq_len, num_variates):
    """
    Determine trigger network configuration based on sequence length and variates.
    Short sequences -> short-term forecasting trigger
    Long sequences -> long-term forecasting trigger
    High variates -> larger model capacity
    """
    if seq_len <= SHORT_SEQ_THRESHOLD:
        # Short sequences: small patch, quick trigger
        config = {
            'pred_len': max(8, seq_len // 8),
            'patch_len': max(4, seq_len // 8),
            'stride': max(2, seq_len // 16),
            'd_model_bd': 16,
            'd_ff_bd': 32,
            'e_layers_bd': 1,
            'forecast_type': 'short'
        }
    elif seq_len <= MEDIUM_SEQ_THRESHOLD:
        # Medium sequences: balanced trigger
        config = {
            'pred_len': seq_len // 4,
            'patch_len': 16,
            'stride': 8,
            'd_model_bd': 32,
            'd_ff_bd': 64,
            'e_layers_bd': 1,
            'forecast_type': 'medium'
        }
    else:
        # Long sequences: long-term forecasting trigger
        config = {
            'pred_len': min(seq_len // 2, 256),
            'patch_len': 32,
            'stride': 16,
            'd_model_bd': 64,
            'd_ff_bd': 128,
            'e_layers_bd': 2,
            'forecast_type': 'long'
        }
    
    # Scale trigger model based on number of variates
    # The trigger must handle all input channels, so high-dimensional data needs more capacity
    if num_variates > 100:
        # Very high-dimensional (e.g., PEMS-SF with 963 variates)
        config['d_model_bd'] = min(config['d_model_bd'] * 4, 256)
        config['d_ff_bd'] = min(config['d_ff_bd'] * 4, 512)
        config['e_layers_bd'] = max(config['e_layers_bd'], 2)
        config['n_heads_bd'] = 8
    elif num_variates > 50:
        # High-dimensional (e.g., MotorImagery with 64 variates, Heartbeat with 61)
        config['d_model_bd'] = min(config['d_model_bd'] * 2, 128)
        config['d_ff_bd'] = min(config['d_ff_bd'] * 2, 256)
        config['n_heads_bd'] = 4
    elif num_variates > 10:
        # Moderate variates
        config['d_model_bd'] = min(int(config['d_model_bd'] * 1.5), 96)
        config['d_ff_bd'] = min(int(config['d_ff_bd'] * 1.5), 192)
        config['n_heads_bd'] = 4
    else:
        # Low variates - default heads
        config['n_heads_bd'] = 2
    
    return config

def get_training_epochs(num_train, seq_len):
    """
    Determine training epochs based on dataset size.
    
    Rationale:
    - Clean training: Small datasets need more epochs to converge, large datasets converge faster
    - BD training: Similar logic for poisoned model training
    - Trigger training: Keep relatively fixed - the trigger needs enough epochs to learn
      a generalizable perturbation regardless of dataset size. We use slightly more for
      small datasets to compensate for fewer gradient updates per epoch.
    """
    if num_train < SMALL_DATASET_THRESHOLD:
        train_epochs = 30
        bd_epochs = 30
        trigger_epochs = 20  # More epochs for small data (fewer updates per epoch)
    elif num_train < 1000:
        train_epochs = 30
        bd_epochs = 30
        trigger_epochs = 15  # Moderate epochs
    else:
        train_epochs = 20
        bd_epochs = 20
        trigger_epochs = 15  # Keep same - large data doesn't mean easier trigger learning
    
    # Adjust for long sequences (more epochs needed for complex patterns)
    if seq_len > 1000:
        train_epochs = int(train_epochs * 1.5)
        bd_epochs = int(bd_epochs * 1.5)
        trigger_epochs = int(trigger_epochs * 1.2)  # Slightly more for complex sequences
    
    return train_epochs, bd_epochs, trigger_epochs

def generate_clean_script(dataset, model):
    """Generate clean training script for a dataset."""
    name = dataset['name']
    seq_len = dataset['seq_len']
    num_train = dataset['num_train']
    num_variates = dataset['num_variates']
    
    batch_size = get_batch_size(num_train, seq_len)
    d_model, d_ff = get_model_config(seq_len, num_variates)
    train_epochs, _, _ = get_training_epochs(num_train, seq_len)
    
    script = f'''python -u main.py --mode clean --model {model} --root_path ./dataset/{name} --seq_len {seq_len} --batch_size {batch_size} --d_model {d_model} --d_ff {d_ff} --train_epochs {train_epochs} --lr 0.001 --gpu_id cuda:0
'''
    return script

def generate_backdoor_script(dataset, model, tmodel, mode='basic'):
    """Generate backdoor training script for a dataset."""
    name = dataset['name']
    seq_len = dataset['seq_len']
    num_train = dataset['num_train']
    num_variates = dataset['num_variates']
    num_classes = dataset['num_classes']
    
    batch_size = get_batch_size(num_train, seq_len)
    d_model, d_ff = get_model_config(seq_len, num_variates)
    train_epochs, bd_epochs, trigger_epochs = get_training_epochs(num_train, seq_len)
    trigger_config = get_trigger_config(seq_len, num_variates)
    
    # Target label: use class 0 or middle class for variety
    target_label = 0
    
    # Poisoning ratio: higher for small datasets
    if num_train < SMALL_DATASET_THRESHOLD:
        poisoning_ratio = 0.1
    else:
        poisoning_ratio = 0.1
    
    script = f'''python -u main.py --mode {mode} --model {model} --Tmodel {tmodel} --root_path ./dataset/{name} --seq_len {seq_len} --batch_size {batch_size} --d_model {d_model} --d_ff {d_ff} --d_model_bd {trigger_config['d_model_bd']} --d_ff_bd {trigger_config['d_ff_bd']} --e_layers_bd {trigger_config['e_layers_bd']} --n_heads_bd {trigger_config['n_heads_bd']} --patch_len {trigger_config['patch_len']} --stride {trigger_config['stride']} --train_epochs {train_epochs} --bd_train_epochs {bd_epochs} --trigger_epochs {trigger_epochs} --lr 0.001 --trigger_lr 0.001 --target_label {target_label} --poisoning_ratio {poisoning_ratio} --clip_ratio 0.1 --gpu_id cuda:0
'''
    return script, trigger_config['forecast_type']

def main():
    # Create output directories
    os.makedirs(os.path.join(script_dir, 'backdoor'), exist_ok=True)
    os.makedirs(os.path.join(script_dir, 'clean'), exist_ok=True)
    
    models = ['TimesNet', 'lstm', 'iTransformer', 'TimeMixer','resnet','mlp','nonstationary_transformer','TCN','BiRNN']
    trigger_models = ['patchtst', 'itst']
    
    # Generate summary info
    summary_lines = ['# Dataset Script Generation Summary\n\n']
    summary_lines.append('| Dataset | Seq Len | Variates | Type | Batch | d_model | Trigger Config (d_bd, ff_bd, heads) |\n')
    summary_lines.append('|---------|---------|----------|------|-------|---------|-------------------------------------|\n')
    
    # === CLEAN TRAINING SCRIPTS ===
    for model in models:
        script_content = f'#!/bin/bash\n# Clean training scripts for {model}\n'
        script_content += f'# Generated for all datasets based on dataset_info.csv\n\n'
        
        for ds in datasets:
            script_content += f'# === {ds["name"]} ===\n'
            script_content += f'# Seq: {ds["seq_len"]}, Vars: {ds["num_variates"]}, Classes: {ds["num_classes"]}, Train: {ds["num_train"]}\n'
            script_content += generate_clean_script(ds, model)
            script_content += '\n'
        
        with open(os.path.join(script_dir, 'clean', f'{model}_all_datasets.sh'), 'w', newline='\n') as f:
            f.write(script_content)
    
    # === BACKDOOR TRAINING SCRIPTS ===
    for model in models:
        for tmodel in trigger_models:
            for mode in ['basic', 'marksman']:
                script_content = f'#!/bin/bash\n# Backdoor training: {model} + {tmodel} trigger ({mode} mode)\n'
                script_content += f'# Trigger networks perform forecasting-based perturbations\n\n'
                
                for ds in datasets:
                    script, forecast_type = generate_backdoor_script(ds, model, tmodel, mode)
                    script_content += f'# === {ds["name"]} ({forecast_type}-term forecasting trigger) ===\n'
                    script_content += f'# Seq: {ds["seq_len"]}, Vars: {ds["num_variates"]}, Classes: {ds["num_classes"]}\n'
                    script_content += script
                    script_content += '\n'
                
                filename = os.path.join(script_dir, 'backdoor', f'{model}_{tmodel}_{mode}_all_datasets.sh')
                with open(filename, 'w', newline='\n') as f:
                    f.write(script_content)
    
    # === INDIVIDUAL DATASET SCRIPTS ===
    os.makedirs(os.path.join(script_dir, 'by_dataset'), exist_ok=True)
    
    for ds in datasets:
        trigger_config = get_trigger_config(ds['seq_len'], ds['num_variates'])
        batch_size = get_batch_size(ds['num_train'], ds['seq_len'])
        d_model, d_ff = get_model_config(ds['seq_len'], ds['num_variates'])
        
        summary_lines.append(
            f'| {ds["name"]} | {ds["seq_len"]} | {ds["num_variates"]} | {trigger_config["forecast_type"]} | '
            f'{batch_size} | {d_model} | d_bd={trigger_config["d_model_bd"]}, ff_bd={trigger_config["d_ff_bd"]}, heads={trigger_config["n_heads_bd"]} |\n'
        )
        
        script_content = f'#!/bin/bash\n# All experiments for {ds["name"]}\n'
        script_content += f'# Sequence Length: {ds["seq_len"]} -> {trigger_config["forecast_type"]}-term forecasting trigger\n'
        script_content += f'# Variates: {ds["num_variates"]}, Classes: {ds["num_classes"]}\n'
        script_content += f'# Train/Test: {ds["num_train"]}/{ds["num_test"]}\n\n'
        
        # Clean training for each model
        '''
                script_content += '# ========== CLEAN TRAINING ==========\n'
        for model in models:
            script_content += f'\n# {model} Clean\n'
            script_content += generate_clean_script(ds, model)
        '''
        
        # Backdoor training for each combination
        script_content += '\n# ========== BACKDOOR TRAINING (BASIC MODE) ==========\n'
        for model in models:
            for tmodel in trigger_models:
                script_content += f'\n# {model} + {tmodel} trigger\n'
                script, _ = generate_backdoor_script(ds, model, tmodel, 'basic')
                script_content += script
        
        script_content += '\n# ========== BACKDOOR TRAINING (MARKSMAN MODE) ==========\n'
        for model in models:
            for tmodel in trigger_models:
                script_content += f'\n# {model} + {tmodel} trigger (Marksman)\n'
                script, _ = generate_backdoor_script(ds, model, tmodel, 'marksman')
                script_content += script
        
        with open(os.path.join(script_dir, 'by_dataset', f'{ds["name"]}.sh'), 'w', newline='\n') as f:
            f.write(script_content)
    
    # Write summary
    with open(os.path.join(script_dir, 'SCRIPT_SUMMARY.md'), 'w', encoding='utf-8') as f:
        f.writelines(summary_lines)
        f.write('\n## Trigger Network Forecast Types\n')
        f.write('- **short**: seq_len <= 100 -> pred_len = seq_len/8\n')
        f.write('- **medium**: 100 < seq_len <= 500 -> pred_len = seq_len/4\n')
        f.write('- **long**: seq_len > 500 -> pred_len = min(seq_len/2, 256)\n')
    
    print(f"Generated scripts for {len(datasets)} datasets:")
    print(f"  - scripts/clean/: Clean training scripts by model")
    print(f"  - scripts/backdoor/: Backdoor scripts by model+trigger+mode")
    print(f"  - scripts/by_dataset/: All experiments per dataset")
    print(f"  - scripts/SCRIPT_SUMMARY.md: Configuration summary")

if __name__ == '__main__':
    main()
