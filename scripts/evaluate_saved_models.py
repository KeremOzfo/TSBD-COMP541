"""
Script to evaluate pre-trained trigger models on their respective datasets.

This script:
1. Scans the Saved_models directory for .pth checkpoint files
2. Extracts metadata (dataset, model type, method) from each checkpoint
3. Generates evaluation commands for each model on its original dataset
4. Optionally generates cross-dataset evaluation commands

Usage:
    python scripts/evaluate_saved_models.py --output_dir scripts/evaluation_scripts
    python scripts/evaluate_saved_models.py --cross_dataset  # Also evaluate on other datasets
"""

import os
import torch
import argparse
from pathlib import Path
import json


def extract_metadata_from_checkpoint(checkpoint_path):
    """Extract metadata from a trigger model checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        
    Returns:
        dict with keys: model_type, dataset, method, seq_len, enc_in, etc.
        or None if metadata cannot be extracted
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Try to get metadata from checkpoint
        if isinstance(checkpoint, dict):
            metadata = {
                'model_type': checkpoint.get('model_type', 'unknown'),
                'dataset': checkpoint.get('dataset', 'unknown'),
                'method': checkpoint.get('method', 'basic'),
                'seq_len': checkpoint.get('seq_len', None),
                'enc_in': checkpoint.get('enc_in', None),
                'd_model_bd': checkpoint.get('d_model_bd', None),
            }
            
            # Also get full args if available
            if 'args' in checkpoint:
                metadata['args'] = checkpoint['args']
            
            return metadata
        else:
            # Old format without metadata - try to parse from filename
            filename = Path(checkpoint_path).stem
            # Format: trigger_{Tmodel}_{dataset}_{method}.pth
            parts = filename.split('_')
            if len(parts) >= 4:
                return {
                    'model_type': parts[1],
                    'dataset': parts[2],
                    'method': parts[3],
                    'seq_len': None,
                    'enc_in': None,
                }
            
            return None
            
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None


def generate_evaluation_command(checkpoint_path, metadata, target_model='TimesNet', 
                                base_data_dir='./dataset', output_dir='Results/evaluation'):
    """Generate evaluation command for a trigger model.
    
    Args:
        checkpoint_path: Path to checkpoint file
        metadata: Metadata dict from extract_metadata_from_checkpoint
        target_model: Target classification model to attack
        base_data_dir: Base directory containing datasets
        output_dir: Where to save evaluation results
        
    Returns:
        Command string to evaluate the model
    """
    dataset = metadata['dataset']
    tmodel = metadata['model_type']
    method = metadata['method']
    
    # Construct dataset path with forward slashes (works on both Windows and Unix)
    dataset_path = f"{base_data_dir}/{dataset}"
    # Normalize checkpoint path to use forward slashes
    checkpoint_path = checkpoint_path.replace('\\', '/')
    
    cmd = f"python main.py"
    cmd += f" --train_mode backdoor"
    cmd += f" --use_pretrained_trigger"
    cmd += f" --trigger_model_path {checkpoint_path}"
    cmd += f" --root_path {dataset_path}"
    cmd += f" --model {target_model}"
    cmd += f" --Tmodel {tmodel}"
    cmd += f" --method {method}"
    cmd += f" --save_test_samples"
    # Add default training parameters (won't be used since trigger is pre-trained)
    cmd += f" --trigger_epochs 0"  # No training needed
    cmd += f" --bd_train_epochs 10"
    return cmd


def scan_saved_models(saved_models_dir='Saved_models'):
    """Scan directory for saved model checkpoints and extract metadata.
    
    Args:
        saved_models_dir: Directory containing .pth files
        
    Returns:
        List of (checkpoint_path, metadata) tuples
    """
    saved_models_dir = Path(saved_models_dir)
    if not saved_models_dir.exists():
        print(f"Directory {saved_models_dir} does not exist!")
        return []
    
    models = []
    for pth_file in saved_models_dir.glob('*.pth'):
        metadata = extract_metadata_from_checkpoint(pth_file)
        if metadata:
            models.append((str(pth_file), metadata))
            print(f"Found: {pth_file.name}")
            print(f"  Dataset: {metadata['dataset']}, Model: {metadata['model_type']}, Method: {metadata['method']}")
        else:
            print(f"Warning: Could not extract metadata from {pth_file.name}")
    
    return models


def generate_evaluation_scripts(models, output_dir='scripts/evaluation_scripts', 
                                target_models=['TimesNet'],
                                cross_dataset=False, all_datasets=None):
    """Generate bash/PowerShell scripts to evaluate all models.
    
    Args:
        models: List of (checkpoint_path, metadata) tuples
        output_dir: Where to save generated scripts
        target_models: List of target classification models to test against
        cross_dataset: If True, also generate cross-dataset evaluation commands
        all_datasets: List of all available datasets for cross-dataset testing
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate bash script
    bash_script = output_dir / 'evaluate_all.sh'
    ps_script = output_dir / 'evaluate_all.ps1'
    
    bash_lines = ['#!/bin/bash', '', '# Auto-generated evaluation script', '']
    ps_lines = ['# Auto-generated evaluation script (PowerShell)', '']
    
    # Same-dataset evaluation
    bash_lines.append('echo "=== Evaluating models on their original datasets ==="')
    ps_lines.append('Write-Host "=== Evaluating models on their original datasets ==="')
    
    for checkpoint_path, metadata in models:
        for target_model in target_models:
            cmd = generate_evaluation_command(checkpoint_path, metadata, target_model)
            
            bash_lines.append(f'echo "Evaluating {Path(checkpoint_path).name} on {metadata["dataset"]} with {target_model}"')
            bash_lines.append(cmd)
            bash_lines.append('')
            
            ps_lines.append(f'Write-Host "Evaluating {Path(checkpoint_path).name} on {metadata["dataset"]} with {target_model}"')
            ps_lines.append(cmd)
            ps_lines.append('')
    
    # Cross-dataset evaluation (optional)
    if cross_dataset and all_datasets:
        bash_lines.append('echo "=== Cross-dataset evaluation ==="')
        ps_lines.append('Write-Host "=== Cross-dataset evaluation ==="')
        
        for checkpoint_path, metadata in models:
            source_dataset = metadata['dataset']
            
            # Test on other datasets
            for target_dataset in all_datasets:
                if target_dataset == source_dataset:
                    continue  # Skip same dataset
                
                for target_model in target_models:
                    # Modify dataset path in command
                    cmd = generate_evaluation_command(checkpoint_path, metadata, target_model)
                    cmd = cmd.replace(f"--root_path ./dataset/{source_dataset}", 
                                    f"--root_path ./dataset/{target_dataset}")
                    
                    bash_lines.append(f'echo "Cross-eval: {Path(checkpoint_path).name} ({source_dataset}) on {target_dataset} with {target_model}"')
                    bash_lines.append(cmd)
                    bash_lines.append('')
                    
                    ps_lines.append(f'Write-Host "Cross-eval: {Path(checkpoint_path).name} ({source_dataset}) on {target_dataset} with {target_model}"')
                    ps_lines.append(cmd)
                    ps_lines.append('')
    
    # Write scripts
    with open(bash_script, 'w') as f:
        f.write('\n'.join(bash_lines))
    
    with open(ps_script, 'w') as f:
        f.write('\n'.join(ps_lines))
    
    # Make bash script executable (on Unix-like systems)
    try:
        os.chmod(bash_script, 0o755)
    except:
        pass
    
    print(f"\nGenerated scripts:")
    print(f"  Bash: {bash_script}")
    print(f"  PowerShell: {ps_script}")
    
    # Also generate a summary JSON
    summary = {
        'total_models': len(models),
        'models': [
            {
                'checkpoint': checkpoint_path,
                'dataset': metadata['dataset'],
                'model_type': metadata['model_type'],
                'method': metadata['method'],
            }
            for checkpoint_path, metadata in models
        ]
    }
    
    summary_file = output_dir / 'evaluation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Summary: {summary_file}")
    print(f"\nTotal commands: {len(models) * len(target_models)}")


def main():
    parser = argparse.ArgumentParser(description='Generate evaluation scripts for saved trigger models')
    parser.add_argument('--saved_models_dir', type=str, default='Saved_models',
                       help='Directory containing saved .pth model files')
    parser.add_argument('--output_dir', type=str, default='scripts/evaluation_scripts',
                       help='Directory to save generated scripts')
    parser.add_argument('--target_models', type=str, nargs='+', default=['TCN'],
                       help='Target classification models to evaluate against')
    parser.add_argument('--cross_dataset', action='store_true',
                       help='Also generate cross-dataset evaluation commands')
    parser.add_argument('--all_datasets', type=str, nargs='+',
                       default=['BasicMotions', 'Epilepsy', 'UWaveGestureLibrary', 
                               'ElectricDevices', 'JapaneseVowels'],
                       help='List of all available datasets for cross-dataset testing')
    
    args = parser.parse_args()
    
    print("Scanning for saved models...")
    models = scan_saved_models(args.saved_models_dir)
    
    if not models:
        print("No models found!")
        return
    
    print(f"\nFound {len(models)} model(s)")
    print("\nGenerating evaluation scripts...")
    
    generate_evaluation_scripts(
        models,
        output_dir=args.output_dir,
        target_models=args.target_models,
        cross_dataset=args.cross_dataset,
        all_datasets=args.all_datasets
    )
    
    print("\nDone! Run the generated scripts to evaluate models:")
    print(f"  Linux/Mac: bash {args.output_dir}/evaluate_all.sh")
    print(f"  Windows:   powershell {args.output_dir}/evaluate_all.ps1")


if __name__ == '__main__':
    main()
