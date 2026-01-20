#!/usr/bin/env python3
"""
Comprehensive backdoor experiment plotting script.

Takes:
1. CSV/Excel from eval_results.py (backdoor results with CA, ASR, method, model, Tmodel, root_path/dataset)
2. JSONL file with clean baseline results (dataset, model, final_acc)

Generates multiple bar plots:
- Method comparison (ASR/CA by method, averaged across datasets/models)
- Tmodel comparison (ASR by trigger model)
- Model robustness (which model has lowest ASR on average)
- Dataset difficulty (which datasets are hardest to backdoor)
- Clean accuracy drop (CA after backdoor vs clean baseline)
- Heatmaps for detailed breakdowns

All figures are saved to a specified output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_backdoor_results(csv_path: Path) -> pd.DataFrame:
    """Load backdoor results from CSV or Excel."""
    if csv_path.suffix == '.xlsx':
        df = pd.read_excel(csv_path)
    else:
        # Try different delimiters and decimal separators
        try:
            df = pd.read_csv(csv_path, sep=';', decimal=',')
        except Exception:
            try:
                df = pd.read_csv(csv_path, sep=',')
            except Exception:
                df = pd.read_csv(csv_path, sep='\t')
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Rename root_path to dataset if needed
    if 'root_path' in df.columns and 'dataset' not in df.columns:
        df['dataset'] = df['root_path']
    
    # Abbreviate long model names for plotting
    if 'model' in df.columns:
        df['model'] = df['model'].replace('nonstationary_transformer', 'NST')
    if 'tmodel' in df.columns:
        df['tmodel'] = df['tmodel'].replace('nonstationary_transformer', 'NST')
    
    # Rename methods for better display
    if 'method' in df.columns:
        df['method'] = df['method'].replace('pureinputaware', 'InputAware')
    
    # Ensure numeric
    df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
    df['asr'] = pd.to_numeric(df['asr'], errors='coerce')
    
    return df


def load_clean_results(jsonl_path: Path) -> pd.DataFrame:
    """Load clean results from JSONL file."""
    records = []
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    df = pd.DataFrame.from_records(records)
    
    # Average duplicates (same dataset + model)
    if not df.empty and 'dataset' in df.columns and 'model' in df.columns:
        df = df.groupby(['dataset', 'model'], as_index=False).agg({
            'final_acc': 'mean',
            'best_acc': 'mean' if 'best_acc' in df.columns else 'first'
        })
    
    return df


def aggregate_duplicates(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Average CA/ASR for duplicate parameter combinations."""
    valid_cols = [c for c in group_cols if c in df.columns]
    if not valid_cols:
        return df
    
    agg_df = df.groupby(valid_cols, dropna=False).agg({
        'ca': 'mean',
        'asr': 'mean',
    }).reset_index()
    
    return agg_df


def save_plot(save_dir: Path, name: str):
    """Save current plot as both PNG and PDF."""
    png_path = save_dir / f"{name}.png"
    pdf_path = save_dir / f"{name}.pdf"
    
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_method_comparison(df: pd.DataFrame, save_dir: Path):
    """Bar plot comparing methods (basic, marksman, etc.) by ASR and CA."""
    if 'method' not in df.columns:
        print("No 'method' column found, skipping method comparison")
        return
    
    method_stats = df.groupby('method').agg({
        'asr': 'mean',
        'ca': 'mean'
    }).reset_index()
    method_stats.columns = ['method', 'asr_mean', 'ca_mean']
    method_stats = method_stats.sort_values('asr_mean', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ASR by method
    ax = axes[0]
    bars = ax.bar(method_stats['method'], method_stats['asr_mean'] * 100, 
                  color=sns.color_palette("husl", len(method_stats)))
    ax.set_xlabel('Attack Method', fontsize=12)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax.set_title('ASR by Attack Method (Higher = More Effective Attack)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, method_stats['asr_mean']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val*100:.1f}%', 
                ha='center', va='bottom', fontsize=10)
    
    # CA by method
    ax = axes[1]
    bars = ax.bar(method_stats['method'], method_stats['ca_mean'] * 100,
                  color=sns.color_palette("husl", len(method_stats)))
    ax.set_xlabel('Attack Method', fontsize=12)
    ax.set_ylabel('Clean Accuracy (%)', fontsize=12)
    ax.set_title('CA by Attack Method (Higher = Better Utility Preservation)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, method_stats['ca_mean']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val*100:.1f}%', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_plot(save_dir, 'method_comparison')
    plt.close()


def plot_tmodel_comparison(df: pd.DataFrame, save_dir: Path):
    """Bar plot comparing trigger models by ASR."""
    if 'tmodel' not in df.columns:
        print("No 'tmodel' column found, skipping Tmodel comparison")
        return
    
    tmodel_stats = df.groupby('tmodel').agg({
        'asr': 'mean',
        'ca': 'mean'
    }).reset_index()
    tmodel_stats.columns = ['tmodel', 'asr_mean', 'ca_mean']
    tmodel_stats = tmodel_stats.sort_values('asr_mean', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = sns.color_palette("Set2", len(tmodel_stats))
    
    # ASR by Tmodel
    ax = axes[0]
    bars = ax.bar(tmodel_stats['tmodel'], tmodel_stats['asr_mean'] * 100,
                  color=colors)
    ax.set_xlabel('Trigger Model', fontsize=12)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax.set_title('ASR by Trigger Model', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, tmodel_stats['asr_mean']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val*100:.1f}%', 
                ha='center', va='bottom', fontsize=10)
    
    # CA by Tmodel
    ax = axes[1]
    bars = ax.bar(tmodel_stats['tmodel'], tmodel_stats['ca_mean'] * 100,
                  color=colors)
    ax.set_xlabel('Trigger Model', fontsize=12)
    ax.set_ylabel('Clean Accuracy (%)', fontsize=12)
    ax.set_title('CA by Trigger Model', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, tmodel_stats['ca_mean']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val*100:.1f}%', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_plot(save_dir, 'tmodel_comparison')
    plt.close()


def plot_model_robustness(df: pd.DataFrame, save_dir: Path):
    """Bar plot showing model robustness (lower ASR = more robust)."""
    if 'model' not in df.columns:
        print("No 'model' column found, skipping model robustness")
        return
    
    model_stats = df.groupby('model').agg({
        'asr': 'mean',
        'ca': 'mean'
    }).reset_index()
    model_stats.columns = ['model', 'asr_mean', 'ca_mean']
    model_stats = model_stats.sort_values('asr_mean', ascending=True)  # Lower ASR = more robust
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = sns.color_palette("coolwarm_r", len(model_stats))
    bars = ax.barh(model_stats['model'], model_stats['asr_mean'] * 100,
                   color=colors)
    
    ax.set_xlabel('Attack Success Rate (%)', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title('Model Robustness Against Backdoor Attacks\n(Lower ASR = More Robust)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 105)
    
    for bar, val in zip(bars, model_stats['asr_mean']):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                f'{val*100:.1f}%', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    save_plot(save_dir, 'model_robustness')
    plt.close()


def plot_dataset_difficulty(df: pd.DataFrame, save_dir: Path):
    """Bar plot showing which datasets are hardest to backdoor (lowest ASR)."""
    if 'dataset' not in df.columns:
        print("No 'dataset' column found, skipping dataset difficulty")
        return
    
    dataset_stats = df.groupby('dataset').agg({
        'asr': 'mean',
        'ca': 'mean'
    }).reset_index()
    dataset_stats.columns = ['dataset', 'asr_mean', 'ca_mean']
    dataset_stats = dataset_stats.sort_values('asr_mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(dataset_stats) * 0.4)))
    
    colors = sns.color_palette("RdYlGn_r", len(dataset_stats))
    bars = ax.barh(dataset_stats['dataset'], dataset_stats['asr_mean'] * 100,
                   color=colors)
    
    ax.set_xlabel('Attack Success Rate (%)', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    ax.set_title('Dataset Difficulty for Backdoor Attacks\n(Lower ASR = Harder to Attack)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 105)
    
    for bar, val in zip(bars, dataset_stats['asr_mean']):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{val*100:.1f}%', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    save_plot(save_dir, 'dataset_difficulty')
    plt.close()


def plot_ca_drop(df: pd.DataFrame, clean_df: pd.DataFrame, save_dir: Path):
    """Bar plot showing clean accuracy drop after backdoor attack."""
    if clean_df.empty:
        print("No clean results provided, skipping CA drop plot")
        return
    
    if 'dataset' not in df.columns or 'model' not in df.columns:
        print("Missing dataset/model columns, skipping CA drop plot")
        return
    
    # Average backdoor CA per dataset+model
    bd_ca = df.groupby(['dataset', 'model'])['ca'].mean().reset_index()
    bd_ca.columns = ['dataset', 'model', 'bd_ca']
    
    # Merge with clean results
    clean_df_renamed = clean_df[['dataset', 'model', 'final_acc']].copy()
    clean_df_renamed.columns = ['dataset', 'model', 'clean_ca']
    
    merged = bd_ca.merge(clean_df_renamed, on=['dataset', 'model'], how='inner')
    merged['ca_drop'] = (merged['clean_ca'] - merged['bd_ca']) * 100
    
    if merged.empty:
        print("No matching dataset/model pairs between backdoor and clean results")
        return
    
    # Aggregate by model
    model_drop = merged.groupby('model').agg({
        'ca_drop': 'mean',
        'clean_ca': 'mean',
        'bd_ca': 'mean'
    }).reset_index()
    model_drop.columns = ['model', 'drop_mean', 'clean_ca', 'bd_ca']
    model_drop = model_drop.sort_values('drop_mean', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # CA drop by model
    ax = axes[0]
    colors = sns.color_palette("Reds", len(model_drop))
    bars = ax.bar(model_drop['model'], model_drop['drop_mean'],
                  color=colors)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title('Clean Accuracy Drop After Backdoor Attack\n(Lower = Better Utility)', 
                 fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, model_drop['drop_mean']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', 
                ha='center', va='bottom', fontsize=10)
    
    # Clean vs Backdoor CA comparison
    ax = axes[1]
    x = np.arange(len(model_drop))
    width = 0.35
    bars1 = ax.bar(x - width/2, model_drop['clean_ca'] * 100, width, label='Clean', color='forestgreen')
    bars2 = ax.bar(x + width/2, model_drop['bd_ca'] * 100, width, label='After Backdoor', color='indianred')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Clean vs Backdoor Clean Accuracy by Model', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_drop['model'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    save_plot(save_dir, 'ca_drop')
    plt.close()


def plot_ca_vs_clean_baseline(df: pd.DataFrame, clean_df: pd.DataFrame, save_dir: Path):
    """Bar plot comparing backdoor CA against clean baseline CA by model."""
    if clean_df.empty:
        print("No clean results provided, skipping CA vs baseline plot")
        return
    
    if 'dataset' not in df.columns or 'model' not in df.columns:
        print("Missing dataset/model columns, skipping CA vs baseline plot")
        return
    
    # Average backdoor CA per dataset+model (same logic as plot_ca_drop)
    bd_ca = df.groupby(['dataset', 'model'])['ca'].mean().reset_index()
    bd_ca.columns = ['dataset', 'model', 'bd_ca']
    
    # Merge with clean results
    clean_df_renamed = clean_df[['dataset', 'model', 'final_acc']].copy()
    clean_df_renamed.columns = ['dataset', 'model', 'clean_ca']
    
    merged = bd_ca.merge(clean_df_renamed, on=['dataset', 'model'], how='inner')
    
    if merged.empty:
        print("No matching dataset/model pairs between backdoor and clean results")
        return
    
    # Aggregate by model
    model_stats = merged.groupby('model').agg({
        'clean_ca': 'mean',
        'bd_ca': 'mean'
    }).reset_index()
    
    model_stats = model_stats.sort_values('clean_ca', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_stats))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, model_stats['clean_ca'] * 100, width, 
                   label='Clean Baseline', color='forestgreen', alpha=0.8)
    bars2 = ax.bar(x + width/2, model_stats['bd_ca'] * 100, width, 
                   label='Backdoor CA', color='indianred', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Clean Accuracy (%)', fontsize=12)
    ax.set_title('Backdoor CA vs Clean Baseline by Model', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_stats['model'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_plot(save_dir, 'ca_vs_clean_baseline')
    plt.close()


def plot_heatmap_method_model(df: pd.DataFrame, save_dir: Path):
    """Heatmap of ASR by method x model (excludes clean method)."""
    if 'method' not in df.columns or 'model' not in df.columns:
        return
    
    # Filter out clean method
    df_filtered = df[df['method'].isin(['basic', 'marksman', 'InputAware'])]
    if df_filtered.empty:
        print("No basic/marksman/InputAware method data for method x model heatmap")
        return
    
    pivot = df_filtered.groupby(['method', 'model'])['asr'].mean().unstack(fill_value=np.nan) * 100
    
    if pivot.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, vmin=0, vmax=100,
                cbar_kws={'label': 'ASR (%)'})
    ax.set_title('ASR (%) by Method × Model', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Method')
    
    plt.tight_layout()
    save_plot(save_dir, 'heatmap_method_model')
    plt.close()


def plot_heatmap_tmodel_model(df: pd.DataFrame, save_dir: Path):
    """Heatmap of ASR by Tmodel x model (excludes clean method)."""
    if 'tmodel' not in df.columns or 'model' not in df.columns:
        return
    
    # Filter out clean method
    df_filtered = df[df['method'].isin(['basic', 'marksman', 'InputAware'])] if 'method' in df.columns else df
    if df_filtered.empty:
        return
    
    pivot = df_filtered.groupby(['tmodel', 'model'])['asr'].mean().unstack(fill_value=np.nan) * 100
    
    if pivot.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, vmin=0, vmax=100,
                cbar_kws={'label': 'ASR (%)'})
    ax.set_title('ASR (%) by Trigger Model × Target Model', fontsize=13, fontweight='bold')
    ax.set_xlabel('Target Model')
    ax.set_ylabel('Trigger Model')
    
    plt.tight_layout()
    save_plot(save_dir, 'heatmap_tmodel_model')
    plt.close()


def plot_heatmap_method_dataset(df: pd.DataFrame, save_dir: Path):
    """Heatmap of ASR by method x dataset (excludes clean method)."""
    if 'method' not in df.columns or 'dataset' not in df.columns:
        return
    
    # Filter out clean method
    df_filtered = df[df['method'].isin(['basic', 'marksman', 'InputAware'])]
    if df_filtered.empty:
        print("No basic/marksman/InputAware method data for method x dataset heatmap")
        return
    
    pivot = df_filtered.groupby(['method', 'dataset'])['asr'].mean().unstack(fill_value=np.nan) * 100
    
    if pivot.empty:
        return
    
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.8), 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, vmin=0, vmax=100,
                cbar_kws={'label': 'ASR (%)'})
    ax.set_title('ASR (%) by Method × Dataset', fontsize=13, fontweight='bold')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Method')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    save_plot(save_dir, 'heatmap_method_dataset')
    plt.close()


def plot_heatmap_tmodel_dataset(df: pd.DataFrame, save_dir: Path):
    """Heatmap of ASR by Tmodel x dataset (excludes clean method)."""
    if 'tmodel' not in df.columns or 'dataset' not in df.columns:
        return
    
    # Filter out clean method
    df_filtered = df[df['method'].isin(['basic', 'marksman', 'InputAware'])] if 'method' in df.columns else df
    if df_filtered.empty:
        return
    
    pivot = df_filtered.groupby(['tmodel', 'dataset'])['asr'].mean().unstack(fill_value=np.nan) * 100
    
    if pivot.empty:
        return
    
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.8), 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, vmin=0, vmax=100,
                cbar_kws={'label': 'ASR (%)'})
    ax.set_title('ASR (%) by Trigger Model × Dataset', fontsize=13, fontweight='bold')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Trigger Model')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    save_plot(save_dir, 'heatmap_tmodel_dataset')
    plt.close()


def plot_heatmap_dataset_model(df: pd.DataFrame, save_dir: Path):
    """Heatmap of ASR by dataset x model."""
    if 'dataset' not in df.columns or 'model' not in df.columns:
        return
    
    pivot = df.groupby(['dataset', 'model'])['asr'].mean().unstack(fill_value=np.nan) * 100
    
    if pivot.empty:
        return
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, vmin=0, vmax=100,
                cbar_kws={'label': 'ASR (%)'})
    ax.set_title('ASR (%) by Dataset × Model', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Dataset')
    
    plt.tight_layout()
    save_plot(save_dir, 'heatmap_dataset_model')
    plt.close()


def plot_best_tmodel_per_model_dataset(df: pd.DataFrame, save_dir: Path):
    """Heatmap showing best Tmodel for each model×dataset combination."""
    if 'tmodel' not in df.columns or 'model' not in df.columns or 'dataset' not in df.columns:
        print("Missing required columns for best Tmodel plot")
        return
    
    # Filter out clean method if exists
    df_filtered = df[df['method'].isin(['basic', 'marksman', 'InputAware'])] if 'method' in df.columns else df
    if df_filtered.empty:
        print("No data for best Tmodel plot")
        return
    
    # For each model×dataset, find the Tmodel with highest ASR
    best_tmodel = df_filtered.loc[df_filtered.groupby(['model', 'dataset'])['asr'].idxmax()]
    
    # Create pivot for display: show Tmodel name with ASR value
    pivot_data = []
    for _, row in best_tmodel.iterrows():
        pivot_data.append({
            'model': row['model'],
            'dataset': row['dataset'],
            'best_tmodel': f"{row['tmodel']}\n({row['asr']*100:.1f}%)",
            'asr': row['asr'] * 100
        })
    
    if not pivot_data:
        return
    
    pivot_df = pd.DataFrame(pivot_data)
    
    # Create text pivot for annotations
    text_pivot = pivot_df.pivot(index='model', columns='dataset', values='best_tmodel')
    # Create numeric pivot for coloring
    num_pivot = pivot_df.pivot(index='model', columns='dataset', values='asr')
    
    fig, ax = plt.subplots(figsize=(max(12, len(text_pivot.columns) * 1.2), max(8, len(text_pivot) * 0.8)))
    sns.heatmap(num_pivot, annot=text_pivot, fmt='', cmap='YlOrRd', ax=ax, vmin=0, vmax=100,
                cbar_kws={'label': 'ASR (%)'}, annot_kws={'fontsize': 9})
    ax.set_title('Best Trigger Model for Each Target Model × Dataset\n(Shows Tmodel with highest ASR)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Target Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    save_plot(save_dir, 'best_tmodel_per_model_dataset')
    plt.close()


def plot_best_method_per_model_dataset(df: pd.DataFrame, save_dir: Path):
    """Heatmap showing best method for each model×dataset combination."""
    if 'method' not in df.columns or 'model' not in df.columns or 'dataset' not in df.columns:
        print("Missing required columns for best method plot")
        return
    
    # Filter to only include relevant methods
    df_filtered = df[df['method'].isin(['basic', 'marksman', 'InputAware'])]
    if df_filtered.empty:
        print("No basic/marksman/InputAware data for best method plot")
        return
    
    # For each model×dataset, find the method with highest ASR
    best_method = df_filtered.loc[df_filtered.groupby(['model', 'dataset'])['asr'].idxmax()]
    
    # Create pivot for display: show method name with ASR value
    pivot_data = []
    for _, row in best_method.iterrows():
        pivot_data.append({
            'model': row['model'],
            'dataset': row['dataset'],
            'best_method': f"{row['method']}\n({row['asr']*100:.1f}%)",
            'asr': row['asr'] * 100
        })
    
    if not pivot_data:
        return
    
    pivot_df = pd.DataFrame(pivot_data)
    
    # Create text pivot for annotations
    text_pivot = pivot_df.pivot(index='model', columns='dataset', values='best_method')
    # Create numeric pivot for coloring
    num_pivot = pivot_df.pivot(index='model', columns='dataset', values='asr')
    
    fig, ax = plt.subplots(figsize=(max(12, len(text_pivot.columns) * 1.2), max(8, len(text_pivot) * 0.8)))
    sns.heatmap(num_pivot, annot=text_pivot, fmt='', cmap='YlOrRd', ax=ax, vmin=0, vmax=100,
                cbar_kws={'label': 'ASR (%)'}, annot_kws={'fontsize': 9})
    ax.set_title('Best Attack Method for Each Target Model × Dataset\n(Shows method with highest ASR)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Target Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    save_plot(save_dir, 'best_method_per_model_dataset')
    plt.close()


def plot_heatmap_method_dataset_ca_drop(df: pd.DataFrame, clean_df: pd.DataFrame, save_dir: Path):
    """Heatmap of CA drop by method x dataset (excludes clean method)."""
    if clean_df.empty:
        print("No clean results provided, skipping method x dataset CA drop heatmap")
        return
    
    if 'method' not in df.columns or 'dataset' not in df.columns:
        return
    
    # Filter out clean method
    df_filtered = df[df['method'].isin(['basic', 'marksman', 'InputAware'])]
    if df_filtered.empty:
        print("No basic/marksman/InputAware method data for method x dataset CA drop heatmap")
        return
    
    # Get backdoor CA per method+dataset
    bd_ca = df_filtered.groupby(['method', 'dataset', 'model'])['ca'].mean().reset_index()
    
    # Merge with clean results
    clean_ca = clean_df[['dataset', 'model', 'final_acc']].copy()
    clean_ca.columns = ['dataset', 'model', 'clean_ca']
    
    merged = bd_ca.merge(clean_ca, on=['dataset', 'model'], how='inner')
    merged['ca_drop'] = (merged['clean_ca'] - merged['ca']) * 100
    
    if merged.empty:
        print("No matching dataset/model pairs for method x dataset CA drop heatmap")
        return
    
    # Average across models for each method+dataset
    pivot = merged.groupby(['method', 'dataset'])['ca_drop'].mean().unstack(fill_value=np.nan)
    
    if pivot.empty:
        return
    
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.8), 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax, vmin=0, vmax=20,
                cbar_kws={'label': 'CA Drop (%)'})
    ax.set_title('CA Drop (%) by Method × Dataset', fontsize=13, fontweight='bold')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Method')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    save_plot(save_dir, 'heatmap_method_dataset_ca_drop')
    plt.close()


def plot_heatmap_tmodel_dataset_ca_drop(df: pd.DataFrame, clean_df: pd.DataFrame, save_dir: Path):
    """Heatmap of CA drop by Tmodel x dataset (excludes clean method)."""
    if clean_df.empty:
        print("No clean results provided, skipping Tmodel x dataset CA drop heatmap")
        return
    
    if 'tmodel' not in df.columns or 'dataset' not in df.columns:
        return
    
    # Filter out clean method
    df_filtered = df[df['method'].isin(['basic', 'marksman', 'InputAware'])] if 'method' in df.columns else df
    if df_filtered.empty:
        print("No data for Tmodel x dataset CA drop heatmap")
        return
    
    # Get backdoor CA per tmodel+dataset
    bd_ca = df_filtered.groupby(['tmodel', 'dataset', 'model'])['ca'].mean().reset_index()
    
    # Merge with clean results
    clean_ca = clean_df[['dataset', 'model', 'final_acc']].copy()
    clean_ca.columns = ['dataset', 'model', 'clean_ca']
    
    merged = bd_ca.merge(clean_ca, on=['dataset', 'model'], how='inner')
    merged['ca_drop'] = (merged['clean_ca'] - merged['ca']) * 100
    
    if merged.empty:
        print("No matching dataset/model pairs for Tmodel x dataset CA drop heatmap")
        return
    
    # Average across models for each tmodel+dataset
    pivot = merged.groupby(['tmodel', 'dataset'])['ca_drop'].mean().unstack(fill_value=np.nan)
    
    if pivot.empty:
        return
    
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.8), 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax, vmin=0, vmax=20,
                cbar_kws={'label': 'CA Drop (%)'})
    ax.set_title('CA Drop (%) by Trigger Model × Dataset', fontsize=13, fontweight='bold')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Trigger Model')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    save_plot(save_dir, 'heatmap_tmodel_dataset_ca_drop')
    plt.close()


def plot_overall_summary(df: pd.DataFrame, save_dir: Path):
    """Create an overall summary figure with key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall ASR distribution
    ax = axes[0, 0]
    ax.hist(df['asr'].dropna() * 100, bins=20, color='indianred', edgecolor='black', alpha=0.7)
    ax.axvline(df['asr'].mean() * 100, color='darkred', linestyle='--', linewidth=2, 
               label=f'Mean: {df["asr"].mean()*100:.1f}%')
    ax.set_xlabel('Attack Success Rate (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of ASR Across All Experiments')
    ax.legend()
    
    # Overall CA distribution
    ax = axes[0, 1]
    ax.hist(df['ca'].dropna() * 100, bins=20, color='forestgreen', edgecolor='black', alpha=0.7)
    ax.axvline(df['ca'].mean() * 100, color='darkgreen', linestyle='--', linewidth=2,
               label=f'Mean: {df["ca"].mean()*100:.1f}%')
    ax.set_xlabel('Clean Accuracy (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of CA Across All Experiments')
    ax.legend()
    
    # ASR vs CA scatter
    ax = axes[1, 0]
    scatter = ax.scatter(df['ca'] * 100, df['asr'] * 100, alpha=0.6, c='steelblue', edgecolors='navy')
    ax.set_xlabel('Clean Accuracy (%)')
    ax.set_ylabel('Attack Success Rate (%)')
    ax.set_title('Trade-off: Clean Accuracy vs Attack Success')
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    
    # Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    stats = [
        ['Total Experiments', len(df)],
        ['Mean ASR', f'{df["asr"].mean()*100:.1f}%'],
        ['Mean CA', f'{df["ca"].mean()*100:.1f}%'],
        ['Unique Models', df['model'].nunique() if 'model' in df.columns else 'N/A'],
        ['Unique Datasets', df['dataset'].nunique() if 'dataset' in df.columns else 'N/A'],
        ['Unique Methods', df['method'].nunique() if 'method' in df.columns else 'N/A'],
        ['Unique Tmodels', df['tmodel'].nunique() if 'tmodel' in df.columns else 'N/A'],
    ]
    
    table = ax.table(cellText=stats, colLabels=['Metric', 'Value'],
                     loc='center', cellLoc='left', colWidths=[0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    ax.set_title('Summary Statistics', fontsize=13, fontweight='bold', y=0.85)
    
    plt.suptitle('Backdoor Attack Results Overview', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_plot(save_dir, 'overall_summary')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate plots from backdoor experiment results.')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to CSV/Excel file with backdoor results')
    parser.add_argument('--clean', type=str, default=str(Path(__file__).parent / 'clean_results.jsonl'),
                        help='Path to JSONL file with clean baseline results (default: utils/clean_results.jsonl)')
    parser.add_argument('--output', type=str, default=str(Path(__file__).parents[1] / 'plots'),
                        help='Output directory for plots (default: project_root/plots)')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading backdoor results from {csv_path}...")
    df = load_backdoor_results(csv_path)
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    
    # Aggregate duplicates
    group_cols = ['model', 'tmodel', 'method', 'dataset']
    df = aggregate_duplicates(df, group_cols)
    print(f"After aggregation: {len(df)} unique parameter combinations")
    
    # Load clean results if provided
    clean_df = pd.DataFrame()
    if args.clean:
        clean_path = Path(args.clean)
        if clean_path.exists():
            print(f"Loading clean results from {clean_path}...")
            clean_df = load_clean_results(clean_path)
            print(f"Loaded {len(clean_df)} clean result records")
        else:
            print(f"Warning: Clean results file not found at {clean_path}")
            print("CA comparison plots will be skipped.")
    
    # Generate all plots
    print("\nGenerating plots...")
    
    plot_overall_summary(df, save_dir)
    plot_method_comparison(df, save_dir)
    plot_tmodel_comparison(df, save_dir)
    plot_model_robustness(df, save_dir)
    plot_dataset_difficulty(df, save_dir)
    plot_ca_drop(df, clean_df, save_dir)
    plot_ca_vs_clean_baseline(df, clean_df, save_dir)
    plot_heatmap_method_model(df, save_dir)
    plot_heatmap_method_dataset(df, save_dir)
    plot_heatmap_method_dataset_ca_drop(df, clean_df, save_dir)
    plot_heatmap_tmodel_model(df, save_dir)
    plot_heatmap_tmodel_dataset(df, save_dir)
    plot_heatmap_tmodel_dataset_ca_drop(df, clean_df, save_dir)
    plot_heatmap_dataset_model(df, save_dir)
    plot_best_tmodel_per_model_dataset(df, save_dir)
    plot_best_method_per_model_dataset(df, save_dir)
    
    print(f"\nAll plots saved to {save_dir}")


if __name__ == '__main__':
    main()
