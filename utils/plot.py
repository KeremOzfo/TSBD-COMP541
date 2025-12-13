"""
Utility plotting helpers for visualizing backdoor behavior on time-series inputs.

Provided tensors/arrays for clean inputs, triggered inputs, predictions, and
true labels, we select examples where the backdoor succeeds (predicted target
label while ground truth is different) and where it fails, then plot both the
original and triggered signals per variate in a single figure.

Usage example:
    plot_backdoor_cases(
        clean_inputs=batch_x,             # shape [N, T, C]
        triggered_inputs=bd_batch_x,      # shape [N, T, C]
        predictions=preds,                # shape [N]
        true_labels=labels,               # shape [N]
        target_label=args.target_label,
        save_dir="plots/backdoor",
        max_success=4,
        max_failure=4,
    )
"""

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy on CPU without gradients."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _select_indices(predictions: np.ndarray, true_labels: np.ndarray, target_label: int,
                    max_success: int, max_failure: int) -> tuple[list[int], list[int]]:
    """Pick indices for successful and failed backdoor attempts."""
    success_mask = (predictions == target_label) & (true_labels != target_label)
    failure_mask = (predictions != target_label) & (true_labels != target_label)

    success_idx = list(np.flatnonzero(success_mask)[:max_success])
    failure_idx = list(np.flatnonzero(failure_mask)[:max_failure])
    return success_idx, failure_idx


def _plot_single(case_name: str, clean_sample: np.ndarray, triggered_sample: np.ndarray,
                 save_path: Path, time_axis: Optional[Iterable[int]] = None,
                 linestyle_trigger: str = "--") -> None:
    """Plot all variates of a single sample with clean vs triggered overlay."""
    # Expect shape [T, C]; if transposed assume [C, T]
    if clean_sample.ndim != 2:
        raise ValueError(f"Expected 2D sample [T, C], got shape {clean_sample.shape}")
    if clean_sample.shape != triggered_sample.shape:
        raise ValueError("Clean and triggered samples must share shape")

    time_steps, num_vars = clean_sample.shape
    x_axis = np.arange(time_steps) if time_axis is None else np.asarray(time_axis)

    fig, axes = plt.subplots(num_vars, 1, figsize=(10, 2.2 * num_vars), sharex=True)
    if num_vars == 1:
        axes = [axes]

    for vidx, ax in enumerate(axes):
        ax.plot(x_axis, clean_sample[:, vidx], label="clean", color="C0", linewidth=1.5)
        ax.plot(x_axis, triggered_sample[:, vidx], label="triggered", color="C3",
                linestyle=linestyle_trigger, linewidth=1.5)
        ax.set_ylabel(f"Var {vidx}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    axes[0].set_title(case_name)
    axes[0].legend(loc="upper right")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_backdoor_cases(
    clean_inputs: torch.Tensor | np.ndarray,
    triggered_inputs: torch.Tensor | np.ndarray,
    predictions: torch.Tensor | np.ndarray,
    true_labels: torch.Tensor | np.ndarray,
    target_label: int,
    save_dir: str | Path,
    max_success: int = 4,
    max_failure: int = 2,
    sample_ids: Optional[Iterable[str]] = None,
) -> dict[str, list[Path]]:
    """Plot successful and failed backdoor examples.

    Args:
        clean_inputs: Clean samples, shape [N, T, C].
        triggered_inputs: Triggered samples, shape [N, T, C].
        predictions: Model predictions, shape [N].
        true_labels: Ground-truth labels, shape [N].
        target_label: Label the backdoor tries to force.
        save_dir: Directory to write figures into.
        max_success: Max number of successful backdoor cases to save.
        max_failure: Max number of failed backdoor cases to save.
        sample_ids: Optional iterable of identifiers (len N) to use in filenames.

    Returns:
        Dict with paths for "success" and "failure" plots.
    """
    clean_np = _to_numpy(clean_inputs)
    triggered_np = _to_numpy(triggered_inputs)
    preds_np = _to_numpy(predictions).astype(int)
    labels_np = _to_numpy(true_labels).astype(int)

    if clean_np.shape != triggered_np.shape:
        raise ValueError("clean_inputs and triggered_inputs must have the same shape")
    if clean_np.ndim != 3:
        raise ValueError("Expected input shape [N, T, C]")

    save_root = Path(save_dir)
    ids = list(sample_ids) if sample_ids is not None else [f"idx{idx}" for idx in range(clean_np.shape[0])]
    if len(ids) != clean_np.shape[0]:
        raise ValueError("sample_ids length must match number of samples")

    success_idx, failure_idx = _select_indices(preds_np, labels_np, target_label, max_success, max_failure)

    saved = {"success": [], "failure": []}

    for idx in success_idx:
        fname = save_root / "success" / f"{ids[idx]}_pred{preds_np[idx]}_true{labels_np[idx]}.png"
        _plot_single(
            case_name=f"Success | pred={preds_np[idx]}, true={labels_np[idx]}, target={target_label}",
            clean_sample=clean_np[idx],
            triggered_sample=triggered_np[idx],
            save_path=fname,
            linestyle_trigger="--",
        )
        saved["success"].append(fname)

    for idx in failure_idx:
        fname = save_root / "failure" / f"{ids[idx]}_pred{preds_np[idx]}_true{labels_np[idx]}.png"
        _plot_single(
            case_name=f"Failure | pred={preds_np[idx]}, true={labels_np[idx]}, target={target_label}",
            clean_sample=clean_np[idx],
            triggered_sample=triggered_np[idx],
            save_path=fname,
            linestyle_trigger=":",
        )
        saved["failure"].append(fname)

    return saved
