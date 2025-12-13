"""Lightweight time-series Grad-CAM style saliency for classification models.

This is not true convolutional Grad-CAM; we compute a per-timestep saliency
by using gradient * input aggregated across channels. It works with models
that accept (x, padding_mask, x_dec, x_mark_dec) and return logits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def pad_or_truncate(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Pad with zeros or truncate on time dimension to match seq_len."""
    cur_len = x.shape[0]
    if cur_len < seq_len:
        pad_len = seq_len - cur_len
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
    elif cur_len > seq_len:
        x = x[:seq_len]
    return x


def compute_time_cam(
    model: torch.nn.Module,
    x: torch.Tensor,
    class_idx: int,
    device: torch.device,
    padding_mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Return normalized per-timestep saliency map in [0,1]."""
    model.eval()
    x = x.unsqueeze(0).to(device)  # [1, T, C]
    x.requires_grad_(True)

    # Provide a default padding mask to avoid model-side None handling errors
    if padding_mask is None:
        padding_mask = torch.ones(1, x.shape[1], device=device)
    else:
        padding_mask = padding_mask.to(device)
        if padding_mask.dim() == 1:
            padding_mask = padding_mask.unsqueeze(0)
        # Ensure mask length matches the input length
        if padding_mask.shape[1] < x.shape[1]:
            pad_len = x.shape[1] - padding_mask.shape[1]
            padding_mask = torch.nn.functional.pad(padding_mask, (0, pad_len), value=1.0)
        elif padding_mask.shape[1] > x.shape[1]:
            padding_mask = padding_mask[:, : x.shape[1]]

    logits = model(x, padding_mask, None, None)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if logits is None:
        return np.zeros(x.shape[1], dtype=np.float32)

    score = logits[0, class_idx]
    model.zero_grad(set_to_none=True)
    score.backward()

    grad = x.grad.detach()[0]  # [T, C]
    cam = (grad * x.detach()[0]).abs().sum(dim=1)  # [T]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.cpu().numpy()


def compute_time_cam_map(
    model: torch.nn.Module,
    x: torch.Tensor,
    class_idx: int,
    device: torch.device,
    padding_mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Return a time x channel saliency map normalized to [0,1].

    Computes |grad * input| per timestep and channel to enable spectrogram-like
    visualization across the full input rather than collapsing channels.
    """
    model.eval()
    x = x.unsqueeze(0).to(device)  # [1, T, C]
    x.requires_grad_(True)

    if padding_mask is None:
        padding_mask = torch.ones(1, x.shape[1], device=device)
    else:
        padding_mask = padding_mask.to(device)
        if padding_mask.dim() == 1:
            padding_mask = padding_mask.unsqueeze(0)
        if padding_mask.shape[1] < x.shape[1]:
            pad_len = x.shape[1] - padding_mask.shape[1]
            padding_mask = torch.nn.functional.pad(padding_mask, (0, pad_len), value=1.0)
        elif padding_mask.shape[1] > x.shape[1]:
            padding_mask = padding_mask[:, : x.shape[1]]

    logits = model(x, padding_mask, None, None)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if logits is None:
        return np.zeros((x.shape[1], x.shape[2]), dtype=np.float32)

    score = logits[0, class_idx]
    model.zero_grad(set_to_none=True)
    score.backward()

    grad = x.grad.detach()[0]  # [T, C]
    cam = (grad * x.detach()[0]).abs()  # [T, C]
    cam = cam - cam.min()
    max_val = cam.max()
    if max_val > 0:
        cam = cam / max_val
    return cam.cpu().numpy()


def plot_time_gradcam(
    clean_input: np.ndarray,
    bd_input: np.ndarray,
    clean_cam: np.ndarray,
    bd_cam: np.ndarray,
    target_label: int,
    save_path: Path,
    sample_id: Optional[str] = None,
) -> None:
    """Plot clean vs backdoor inputs with per-timestep saliency overlays."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def _plot(ax, signal, cam, title, overlay_label=None):
        t = np.arange(signal.shape[0])
        ch0 = signal[:, 0] if signal.ndim == 2 else signal
        ax.plot(t, ch0, label=overlay_label or "signal", linewidth=1.4)
        ax.fill_between(t, ch0.min(), ch0.max(), where=cam > 0, color="orange", alpha=0.25, step="pre")
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    sid = f" (id: {sample_id})" if sample_id is not None else ""
    _plot(axes[0], clean_input, clean_cam, f"Clean input + GradCAM{sid}")

    axes[1].plot(np.arange(clean_input.shape[0]), clean_input[:, 0], linestyle="--", color="gray", label="clean (ch0)")
    _plot(axes[1], bd_input, bd_cam, "Backdoor input + GradCAM", overlay_label="backdoor (ch0)")
    axes[1].legend()
    axes[1].text(0.01, 0.95, f"target label: {target_label}", transform=axes[1].transAxes, va="top", ha="left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_time_gradcam_map(
    clean_input: np.ndarray,
    bd_input: np.ndarray,
    clean_cam_map: np.ndarray,
    bd_cam_map: np.ndarray,
    target_label: int,
    save_path: Path,
    sample_id: Optional[str] = None,
) -> None:
    """Plot spectrogram-style heatmaps over time x channels for clean/backdoor inputs."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    sid = f" (id: {sample_id})" if sample_id is not None else ""

    def _imshow(ax, cam_map, title):
        im = ax.imshow(cam_map.T, aspect="auto", origin="lower", cmap="jet")
        ax.set_ylabel("Channel")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _imshow(axes[0], clean_cam_map, f"Clean GradCAM map{sid}")
    _imshow(axes[1], bd_cam_map, f"Backdoor GradCAM map (target {target_label})")
    axes[1].set_xlabel("Time")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
