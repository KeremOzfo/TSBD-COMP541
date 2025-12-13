"""
Logging and visualization utilities for experiments.

- log_result_clean: save args, final metrics, and training curves for clean runs.
- log_all: record args, final metrics, plots for trigger/model training, and
  backdoor examples using plot_backdoor_cases.
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
    plt.savefig(save_path, dpi=200)
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
    exp_dir = save_root / f"{dataset}_G-{trigger_name}_C-{model_name}_{run_hash}"
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

    # Trigger training plots
    if trigger_results:
        losses = {
            "train_loss": np.asarray(trigger_results.get("train_loss", [])),
            "test_loss": np.asarray(trigger_results.get("test_loss", [])),
        }
        losses = {k: v for k, v in losses.items() if v.size > 0}
        if losses:
            _plot_curve(losses, "Trigger Training Loss", "Loss", exp_dir / "trigger_loss.png")

        accs = {
            "train_clean_acc": np.asarray(trigger_results.get("train_clean_acc", [])),
            "train_ASR": np.asarray(trigger_results.get("train_ASR", [])),
            "test_clean_acc": np.asarray(trigger_results.get("test_clean_acc", [])),
            "test_ASR": np.asarray(trigger_results.get("test_ASR", [])),
        }
        accs = {k: v for k, v in accs.items() if v.size > 0}
        if accs:
            _plot_curve(accs, "Trigger Training Accuracy", "Accuracy", exp_dir / "trigger_accuracy.png")

    # Poisoning stage plots (CA/ASR over epochs)
    if model_poison_dic:
        poison_accs = {
            "CA": np.asarray(model_poison_dic.get("CA", [])),
            "ASR": np.asarray(model_poison_dic.get("ASR", [])),
        }
        poison_accs = {k: v for k, v in poison_accs.items() if v.size > 0}
        if poison_accs:
            _plot_curve(poison_accs, "Poisoned Model Metrics", "Accuracy", exp_dir / "poison_metrics.png")

    # Example plots using plot_backdoor_cases and Grad-CAM overlays
    if sample_cases:
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
                for idx in success_indices[:max_gradcam]:
                    # Skip if sample is missing
                    if clean_inputs[idx] is None or triggered_inputs[idx] is None:
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

    # Latent separability plot (optional)
    if model is not None and test_loader is not None:
        try:
            latent_dir = exp_dir / "latent"
            latent_dir.mkdir(parents=True, exist_ok=True)
            plot_latent_separability(
                model=model,
                loader=test_loader,
                args=args,
                trigger_model=trigger_model,
                method=latent_method,
                save_dir=latent_dir,
                max_points=latent_max_points,
                poisoned_loader=poisoned_loader,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[log_all] Failed to plot latent separability: {exc}")

    return exp_dir


