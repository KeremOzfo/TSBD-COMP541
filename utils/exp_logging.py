"""
Logging and visualization utilities for experiments.

- log_result_clean: save args, final metrics, and training curves for clean runs.
- log_all: record args, final metrics, plots for trigger/model training, and
  backdoor examples using plot_backdoor_cases.
- log_final_test_epoch: comprehensive final test with bd_test_with_samples,
  PCA, t-SNE, GradCAM, and all visualizations (uses log_all internally).
"""
import torch
import json
import os
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import matplotlib.pyplot as plt
import numpy as np

from utils.plot import plot_backdoor_cases
from utils.latent_sep import plot_latent_separability
from utils.gradcam import (
    compute_time_cam,
    compute_time_cam_map,
    plot_time_gradcam,
    plot_time_gradcam_map,
    pad_or_truncate,
)


def log_result_clean(
    args: Any,
    final_acc: float,
    train_history: Optional[Dict[str, List[float]]] = None,
    save_dir: str | Path = "Results",
    run_hash: Optional[str] = None,
) -> Path:
    """Log clean training results with args, metrics, and training curves.
    
    Args:
        args: Training arguments
        final_acc: Final test accuracy
        train_history: Dict with keys like 'train_loss', 'test_loss', 'train_acc', 'test_acc'
                       Each value is a list of per-epoch values
        save_dir: Root directory for saving results
        run_hash: Optional hash for run identification
    
    Returns:
        Path to the experiment directory
    """
    save_root = Path(save_dir)
    if run_hash is None:
        hash_input = str(time.time())
        run_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    print(f"Run hash: {run_hash}")

    dataset = os.path.basename(os.path.normpath(getattr(args, "root_path", "dataset")))
    model_name = getattr(args, "model_name", getattr(args, "model", "model"))
    exp_dir = save_root / f"{dataset}_clean_{model_name}_{run_hash}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Write args and final metrics to txt
    args_path = exp_dir / "args_and_results.txt"
    best_acc = max(train_history.get("test_acc", [final_acc])) if train_history else final_acc
    
    with open(args_path, "w", encoding="utf-8") as f:
        f.write(f"Run hash: {run_hash}\n")
        f.write(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Mode: clean\n")
        f.write("\n[Args]\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
        f.write("\n[Final Metrics]\n")
        f.write(f"Final Test Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)\n")
        f.write(f"Best Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)\n")

    # Also append to JSONL for easy aggregation
    jsonl_path = save_root / "clean_results.jsonl"
    row = {
        "run_hash": run_hash,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "model": model_name,
        "final_acc": final_acc,
        "best_acc": best_acc,
        "train_epochs": getattr(args, "train_epochs", None),
        "lr": getattr(args, "lr", None),
        "batch_size": getattr(args, "batch_size", None),
    }
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    # Plot training curves if history provided
    if train_history:
        # Loss curves
        losses = {
            "train_loss": np.asarray(train_history.get("train_loss", [])),
            "test_loss": np.asarray(train_history.get("test_loss", [])),
        }
        losses = {k: v for k, v in losses.items() if v.size > 0}
        if losses:
            _plot_curve(losses, "Clean Training Loss", "Loss", exp_dir / "clean_loss.png")

        # Accuracy curves
        accs = {
            "train_acc": np.asarray(train_history.get("train_acc", [])),
            "test_acc": np.asarray(train_history.get("test_acc", [])),
        }
        accs = {k: v for k, v in accs.items() if v.size > 0}
        if accs:
            _plot_curve(accs, "Clean Training Accuracy", "Accuracy", exp_dir / "clean_accuracy.png")

    print(f"Clean training results saved to: {exp_dir}")
    return exp_dir


def _plot_curve(values_dict: Dict[str, np.ndarray], title: str, ylabel: str, save_path: Path) -> None:
    """Plot multiple curves on one figure."""
    plt.figure(figsize=(8, 4))
    
    # Get the maximum length of all value arrays to set proper x-axis
    max_epochs = max(len(vals) for vals in values_dict.values())
    
    for name, vals in values_dict.items():
        # Use explicit x values (1-based epochs) to ensure integer ticks
        epochs = list(range(1, len(vals) + 1))
        plt.plot(epochs, vals, label=name, linewidth=2)
    
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Ensure x-axis shows integer ticks only
    plt.xticks(range(1, max_epochs + 1))
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pil_kwargs={'optimize': True, 'quality': 85})
    plt.close()


def log_all(
    args: Any,
    trigger_results: Optional[Dict[str, Any]],
    model_poison_dic: Optional[Dict[str, Any]],
    sample_cases: Optional[Dict[str, Any]] = None,
    model: Optional[torch.nn.Module] = None,
    test_loader: Any = None,
    poisoned_loader: Any = None,
    trigger_model: Optional[torch.nn.Module] = None,
    poison_indices: Optional[list] = None,
    latent_method: str = "pca",
    latent_max_points: int = 2000,
    save_dir: str | Path = "Results",
    run_hash: Optional[str] = None,
) -> Path:
    """Log parameters, metrics, plots, and example traces.

    Saves:
    - args key/values to a txt (with run hash at top and final CA/ASR at bottom)
    - trigger training curves (losses and accuracies) if available
    - poisoning stage CA/ASR curves
    - backdoor example plots using plot_backdoor_cases (if sample_cases provided)

    Returns the experiment directory Path.
    """
    save_root = Path(save_dir)
    if run_hash is None:
        hash_input = str(time.time())
        run_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    print(f"Run hash: {run_hash}")

    dataset = os.path.basename(os.path.normpath(getattr(args, "root_path", "dataset")))
    model_name = getattr(args, "model_name", getattr(args, "model", "model"))
    trigger_name = getattr(args, "Tmodel", "trigger")
    method_name = getattr(args, "method", "method") 
    exp_dir = save_root / f"{dataset}_G-{trigger_name}_C-{model_name}_M-{method_name}_{run_hash}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Write args and final metrics
    args_path = exp_dir / "args_and_results.txt"
    final_ca = None
    final_asr = None
    if model_poison_dic:
        final_ca = model_poison_dic.get("CA", [None])[-1]
        final_asr = model_poison_dic.get("ASR", [None])[-1]

    with open(args_path, "w", encoding="utf-8") as f:
        f.write(f"Run hash: {run_hash}\n")
        f.write(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write("\n[Args]\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
        f.write("\n[Final Metrics]\n")
        f.write(f"Final Clean Accuracy (CA): {final_ca}\n")
        f.write(f"Final Attack Success Rate (ASR): {final_asr}\n")

    # Trigger training plots - only train losses (including loss components by type)
    if trigger_results:
        # Main train loss
        main_loss = {"train_loss": np.asarray(trigger_results.get("train_loss", []))}
        main_loss = {k: v for k, v in main_loss.items() if v.size > 0}
        
        # Loss components by type (each method has different losses)
        loss_components = trigger_results.get("loss_components", {})
        if loss_components:
            # Only use train losses from loss_components
            for key, values in loss_components.items():
                if isinstance(values, (list, np.ndarray)):
                    arr = np.asarray(values)
                    if arr.size > 0:
                        main_loss[key] = arr
        
        if main_loss:
            _plot_curve(main_loss, "Trigger Training Loss Components", "Loss", exp_dir / "trigger_loss.png")

    # Poisoning stage plots (CA/ASR over epochs)
    if model_poison_dic:
        poison_accs = {
            "CA": np.asarray(model_poison_dic.get("CA", [])),
            "ASR": np.asarray(model_poison_dic.get("ASR", [])),
        }
        poison_accs = {k: v for k, v in poison_accs.items() if v.size > 0}
        if poison_accs:
            _plot_curve(poison_accs, "Poisoned Model Metrics", "Accuracy", exp_dir / "poison_metrics.png")

    # Save trigger model if enabled
    save_trigger = getattr(args, 'save_trigger_model', True)
    if save_trigger and trigger_model is not None:
        # Extract dataset name from root_path
        dataset_name = Path(args.root_path).name
        tmodel_name = getattr(args, 'Tmodel', 'unknown')
        method_name = getattr(args, 'method', 'basic')
        
        # Create filename: trigger_{Tmodel}_{dataset}_{method}.pth
        trigger_filename = f"trigger_{tmodel_name}_{dataset_name}_{method_name}.pth"
        trigger_path = exp_dir / trigger_filename
        
        # Save model state dict and metadata
        torch.save({
            'model_state_dict': trigger_model.state_dict(),
            'model_type': tmodel_name,
            'dataset': dataset_name,
            'method': method_name,
            'seq_len': getattr(args, 'seq_len', None),
            'enc_in': getattr(args, 'enc_in', None),
            'd_model_bd': getattr(args, 'd_model_bd', None),
            'args': vars(args),  # Save all args for reference
        }, trigger_path)
        print(f"Trigger model saved to: {trigger_path}")


    # Example plots using plot_backdoor_cases and Grad-CAM overlays
    # Only save if save_test_samples is True (default: True)
    save_test_samples = getattr(args, 'save_test_samples', True)
    
    if sample_cases and save_test_samples:
        manifest_path = exp_dir / "examples" / "example_plots.txt"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            clean_inputs = sample_cases.get("clean_inputs")
            triggered_inputs = sample_cases.get("triggered_inputs")
            preds = sample_cases.get("predictions")
            trues = sample_cases.get("true_labels")

            has_samples = (
                clean_inputs is not None
                and triggered_inputs is not None
                and preds is not None
                and trues is not None
                and len(clean_inputs) > 0
            )

            # Ensure all arrays have the same length (handle variable-length datasets)
            if has_samples:
                min_len = min(len(clean_inputs), len(triggered_inputs), len(preds), len(trues))
                if min_len < len(clean_inputs) or min_len < len(triggered_inputs) or min_len < len(preds) or min_len < len(trues):
                    print(f"[log_all] Warning: Truncating sample arrays to common length {min_len}")
                    print(f"  Original lengths - clean: {len(clean_inputs)}, triggered: {len(triggered_inputs)}, preds: {len(preds)}, trues: {len(trues)}")
                    clean_inputs = clean_inputs[:min_len]
                    triggered_inputs = triggered_inputs[:min_len]
                    preds = preds[:min_len]
                    trues = trues[:min_len]
                    if sample_cases.get("sample_ids") is not None:
                        sample_ids_list = list(sample_cases.get("sample_ids"))
                        if len(sample_ids_list) > min_len:
                            sample_cases["sample_ids"] = sample_ids_list[:min_len]

            target_label = sample_cases.get("target_label", getattr(args, "target_label", 0))

            # Stats for diagnostics
            total_samples = len(clean_inputs) if has_samples else 0
            preds_np = np.asarray(preds).ravel() if has_samples else np.asarray([])
            trues_np = np.asarray(trues).ravel() if has_samples else np.asarray([])
            success_mask = (preds_np == target_label) & (trues_np != target_label) if has_samples else np.asarray([])
            success_indices = np.nonzero(success_mask)[0] if has_samples else []
            success_count = int(success_mask.sum()) if has_samples else 0
            failure_count = int(np.sum((preds_np != target_label) & (trues_np != target_label))) if has_samples else 0
            non_target_truth = int(np.sum(trues_np != target_label)) if has_samples else 0

            saved_paths = {"success": [], "failure": []}
            if has_samples:
                saved_paths = plot_backdoor_cases(
                    clean_inputs=clean_inputs,
                    triggered_inputs=triggered_inputs,
                    predictions=preds,
                    true_labels=trues,
                    target_label=target_label,
                    save_dir=exp_dir / "examples",
                    max_success=5,
                    max_failure=3,
                    sample_ids=sample_cases.get("sample_ids"),
                )
            else:
                print("[log_all] No sample cases available to plot.")

            # Grad-CAM overlays for successful backdoor examples
            if has_samples and model is not None and len(success_indices) > 0:
                gradcam_dir = exp_dir / "examples" / "gradcam"
                gradcam_dir.mkdir(parents=True, exist_ok=True)
                max_gradcam = min(3, len(success_indices))
                print(f"[log_all] Generating GradCAM visualizations for {max_gradcam} samples...")
                for idx in success_indices[:max_gradcam]:
                    try:
                        # Skip if index is out of bounds (can happen with variable-length datasets)
                        if idx >= len(clean_inputs) or idx >= len(triggered_inputs):
                            print(f"[log_all] Skipping sample {idx}: out of bounds")
                            continue
                        # Skip if sample is missing
                        if clean_inputs[idx] is None or triggered_inputs[idx] is None:
                            print(f"[log_all] Skipping sample {idx}: missing data")
                            continue

                        clean_x = torch.tensor(clean_inputs[idx]).float()
                        bd_x = torch.tensor(triggered_inputs[idx]).float()

                        seq_len = getattr(args, "seq_len", clean_x.shape[0])
                        clean_x = pad_or_truncate(clean_x, seq_len)
                        bd_x = pad_or_truncate(bd_x, seq_len)

                        device = getattr(args, "device", torch.device("cpu"))
                        clean_cam = compute_time_cam(model, clean_x, target_label, device)
                        bd_cam = compute_time_cam(model, bd_x, target_label, device)
                        clean_cam_map = compute_time_cam_map(model, clean_x, target_label, device)
                        bd_cam_map = compute_time_cam_map(model, bd_x, target_label, device)

                        sample_id = None
                        if sample_cases.get("sample_ids") is not None and idx < len(sample_cases.get("sample_ids")):
                            sample_id = sample_cases.get("sample_ids")[idx]

                        # Save both GradCAM overlay and map visualizations
                        overlay_path = gradcam_dir / f"sample_{idx}_gradcam_overlay.png"
                        plot_time_gradcam(
                            clean_input=clean_x.cpu().numpy(),
                            bd_input=bd_x.cpu().numpy(),
                            clean_cam=clean_cam,
                            bd_cam=bd_cam,
                            target_label=target_label,
                            save_path=overlay_path,
                            sample_id=sample_id,
                        )

                        map_path = gradcam_dir / f"sample_{idx}_gradcam_map.png"
                        plot_time_gradcam_map(
                            clean_input=clean_x.cpu().numpy(),
                            bd_input=bd_x.cpu().numpy(),
                            clean_cam_map=clean_cam_map,
                            bd_cam_map=bd_cam_map,
                            target_label=target_label,
                            save_path=map_path,
                            sample_id=sample_id,
                        )
                        print(f"[log_all] Generated GradCAM for sample {idx}")
                    except Exception as e:
                        print(f"[log_all] Failed to generate GradCAM for sample {idx}: {e}")
                        import traceback
                        traceback.print_exc()
            elif has_samples and model is not None:
                print("[log_all] No successful backdoor samples for GradCAM visualization")
            elif not has_samples:
                print("[log_all] No sample data available for GradCAM")

            # Always write manifest, even if empty
            with open(manifest_path, "w", encoding="utf-8") as mf:
                mf.write(f"Total samples collected: {total_samples}\n")
                mf.write(f"Ground-truth != target: {non_target_truth}\n")
                mf.write(f"Successful backdoor (pred==target, true!=target): {success_count}\n")
                mf.write(f"Failed backdoor (pred!=target, true!=target): {failure_count}\n\n")

                mf.write("Successful backdoor examples:\n")
                for p in saved_paths.get("success", []):
                    mf.write(f"{p}\n")
                mf.write("\nFailed backdoor examples:\n")
                for p in saved_paths.get("failure", []):
                    mf.write(f"{p}\n")
                if not saved_paths.get("success") and not saved_paths.get("failure"):
                    mf.write("(none saved)\n")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[log_all] Failed to plot backdoor cases: {exc}")
            with open(manifest_path, "w", encoding="utf-8") as mf:
                mf.write(f"Plotting failed: {exc}\n")
    elif sample_cases and not save_test_samples:
        print("[log_all] Skipping test sample and GradCAM visualization (save_test_samples=False)")

    # Latent separability plot (optional) - save both PCA and t-SNE
    if model is not None and test_loader is not None:
        try:
            latent_dir = exp_dir / "latent"
            latent_dir.mkdir(parents=True, exist_ok=True)
            for method in ["pca", "tsne"]:
                plot_latent_separability(
                    model=model,
                    loader=test_loader,
                    args=args,
                    trigger_model=trigger_model,
                    method=method,
                    save_dir=latent_dir,
                    max_points=latent_max_points,
                    poisoned_loader=poisoned_loader,
                    poison_indices=poison_indices,
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[log_all] Failed to plot latent separability: {exc}")

    return exp_dir


def log_final_test_epoch(
    model: torch.nn.Module,
    trigger_model: Optional[torch.nn.Module],
    train_loader,
    test_loader,
    poisoned_loader,
    args: Any,
    sample_cases: Optional[Dict[str, Any]] = None,
    poison_indices: Optional[list] = None,
    trigger_results: Optional[Dict[str, Any]] = None,
    model_poison_dic: Optional[Dict[str, Any]] = None,
    save_dir: str | Path = "Results",
    run_hash: Optional[str] = None,
    run_bd_test: bool = True,
) -> Path:
    """Conduct comprehensive final test epoch with all visualizations and metrics.
    
    This function performs a complete final evaluation including:
    - PCA and t-SNE visualization on train set (using poisoned_loader)
    - ASR and CA metrics
    - Sample cases collection (success/failure examples) - via bd_test_with_samples
    - GradCAM visualizations
    - All results logged via log_all
    
    Args:
        model: Trained victim model
        trigger_model: Trained trigger generator
        train_loader: Clean training data loader
        test_loader: Test data loader
        poisoned_loader: Poisoned training data loader (for PCA/t-SNE)
        args: Arguments with all configurations
        sample_cases: Pre-collected sample cases (if None and run_bd_test=True, will collect)
        trigger_results: Trigger training history (optional)
        model_poison_dic: Model poisoning history (optional)
        save_dir: Root directory for saving results
        run_hash: Optional hash for run identification
        run_bd_test: If True and sample_cases is None, will run bd_test_with_samples
    
    Returns:
        Path to the experiment directory
        
    Note:
        To avoid circular imports, you can either:
        1. Pass sample_cases directly (recommended)
        2. Let this function run bd_test_with_samples by setting run_bd_test=True
    """
    print("\n" + "="*70)
    print("FINAL TEST EPOCH - COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # Run comprehensive test with sample collection if needed
    if sample_cases is None and run_bd_test:
        # Import only when needed to avoid circular dependency at module level
        from test import bd_test_with_samples
        
        print("\n[1/2] Running backdoor test with sample collection...")
        test_loss, clean_acc, asr, sample_cases = bd_test_with_samples(
            model=model,
            loader=test_loader,
            args=args,
            trigger_model=trigger_model,
            max_success=8,
            max_failure=8,
        )
        
        print(f"  ✓ Clean Accuracy: {clean_acc*100:.2f}%")
        print(f"  ✓ Attack Success Rate (ASR): {asr*100:.2f}%")
        print(f"  ✓ Collected {len(sample_cases.get('sample_ids', []))} sample cases")
        
        # Update model_poison_dic with final metrics if not provided
        if model_poison_dic is None:
            model_poison_dic = {
                'CA': [clean_acc],
                'ASR': [asr]
            }
        elif isinstance(model_poison_dic, dict):
            # Ensure final values are in the dict
            if 'CA' not in model_poison_dic or len(model_poison_dic['CA']) == 0:
                model_poison_dic['CA'] = [clean_acc]
            if 'ASR' not in model_poison_dic or len(model_poison_dic['ASR']) == 0:
                model_poison_dic['ASR'] = [asr]
    elif sample_cases is not None:
        print("\n[1/2] Using provided sample cases...")
        print(f"  ✓ Received {len(sample_cases.get('sample_ids', []))} sample cases")
    else:
        print("\n[1/2] No sample cases provided and run_bd_test=False, skipping...")
    
    # Log all results including PCA, t-SNE, GradCAM, sample cases
    save_test_samples = getattr(args, 'save_test_samples', True)
    print("\n[2/2] Generating comprehensive visualizations and logs...")
    print("  - PCA and t-SNE plots (on training set)")
    if save_test_samples:
        print("  - Sample backdoor cases (success/failure)")
        print("  - GradCAM heatmaps")
    else:
        print("  - Skipping sample backdoor cases and GradCAM (save_test_samples=False)")
    print("  - Training curves and metrics")
    
    exp_dir = log_all(
        args=args,
        trigger_results=trigger_results,
        model_poison_dic=model_poison_dic,
        sample_cases=sample_cases,
        model=model,
        test_loader=train_loader,  # Use train_loader for PCA/t-SNE as specified
        poisoned_loader=poisoned_loader,
        trigger_model=trigger_model,
        poison_indices=poison_indices,
        latent_method="pca",  # Will generate both PCA and t-SNE in log_all
        latent_max_points=2000,
        save_dir=save_dir,
        run_hash=run_hash,
    )
    
    print("\n" + "="*70)
    print("FINAL TEST EPOCH COMPLETED")
    print(f"Results saved to: {exp_dir}")
    print("="*70 + "\n")
    
    return exp_dir
