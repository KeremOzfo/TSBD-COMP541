"""
Logging and visualization utilities for experiments.

- log_result_clean: append JSONL row for clean runs.
- log_all: record args, final metrics, plots for trigger/model training, and
  backdoor examples using plot_backdoor_cases.
"""

import json
import os
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from utils.plot import plot_backdoor_cases


def log_result_clean(path: str, info: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = dict(info)
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _plot_curve(values_dict: Dict[str, np.ndarray], title: str, ylabel: str, save_path: Path) -> None:
    """Plot multiple curves on one figure."""
    plt.figure(figsize=(8, 4))
    for name, vals in values_dict.items():
        plt.plot(vals, label=name, linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def log_all(
    args: Any,
    trigger_results: Optional[Dict[str, Any]],
    model_poison_dic: Optional[Dict[str, Any]],
    sample_cases: Optional[Dict[str, Any]] = None,
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

    # Example plots using plot_backdoor_cases
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
            success_count = int(np.sum((preds_np == target_label) & (trues_np != target_label))) if has_samples else 0
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

    return exp_dir


