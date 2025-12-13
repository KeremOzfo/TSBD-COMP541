"""
Latent separability visualization for clean vs backdoored samples.

This module extracts latent features (or logits if no latent is exposed) from a
model on clean inputs and the same inputs with triggers applied, then projects
them to 2D (PCA or t-SNE) and plots the result. It also saves a small text
report with basic statistics.
"""

from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

Method = Literal["pca", "tsne"]


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _get_feature(output: torch.Tensor | tuple) -> torch.Tensor:
    """Pick a latent feature from model output; fallback to logits if needed."""
    if isinstance(output, tuple) and len(output) >= 2:
        return output[1]
    return output


def _project(feats: np.ndarray, method: Method) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2).fit_transform(feats)
    if method == "tsne":
        return TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(feats)
    raise ValueError(f"Unknown method: {method}")


def plot_latent_separability(
    model: torch.nn.Module,
    loader,
    args,
    trigger_model: Optional[torch.nn.Module] = None,  # kept for signature compatibility; unused
    method: Method = "pca",
    save_dir: str | Path = "Results/latent",
    max_points: int = 2000,
    poisoned_loader: Optional[object] = None,
) -> dict:
    """Generate a 2D projection scatter plot for clean vs poisoned inputs.

    Expects `poisoned_loader` to contain already-poisoned samples (no trigger application here).
    Returns a dict with paths and simple stats.
    """
    model.eval()

    clean_feats = []
    bd_feats = []
    labels = []
    total_clean_seen = 0
    total_bd_seen = 0
    with torch.no_grad():
        # Clean forward passes
        for batch_x, label, padding_mask in loader:
            batch_x = batch_x.float().to(args.device)
            padding_mask = padding_mask.float().to(args.device)
            label = label.to(args.device)
            total_clean_seen += batch_x.size(0)

            clean_out = model(batch_x, padding_mask, None, None)
            clean_feat = _get_feature(clean_out)

            clean_feats.append(clean_feat.detach())
            labels.append(label.detach())

        if poisoned_loader is None:
            raise RuntimeError("poisoned_loader is required for latent separability (poisoned samples).")

        # Poisoned forward passes (pre-poisoned dataset). Use all provided poisoned samples.
        for batch_x, label, padding_mask in poisoned_loader:
            batch_x = batch_x.float().to(args.device)
            padding_mask = padding_mask.float().to(args.device)
            label = label.to(args.device)

            total_bd_seen += batch_x.size(0)

            bd_out = model(batch_x, padding_mask, None, None)
            bd_feat = _get_feature(bd_out)
            bd_feats.append(bd_feat.detach())

    if not clean_feats:
        raise RuntimeError("No clean features collected; check the loader or model output.")
    if not bd_feats:
        raise RuntimeError("No poisoned features collected; provide poisoned_loader or check trigger path.")

    clean_feats = torch.cat(clean_feats, dim=0)
    bd_feats = torch.cat(bd_feats, dim=0)
    labels = torch.cat(labels, dim=0).flatten()

    # Downsample for plotting if too large
    def _downsample(feats: torch.Tensor, max_n: int) -> torch.Tensor:
        if feats.shape[0] <= max_n:
            return feats
        idx = torch.randperm(feats.shape[0])[:max_n]
        return feats[idx]

    clean_plot = _downsample(clean_feats, max_points)
    bd_plot = _downsample(bd_feats, max_points)

    # Project jointly so both sets share the same projection space
    combined = torch.cat([clean_plot, bd_plot], dim=0)
    proj_all = _project(_to_numpy(combined), method)
    proj_clean = proj_all[: clean_plot.shape[0]]
    proj_bd = proj_all[clean_plot.shape[0] :]

    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    fig_path = save_root / f"latent_{method}.png"
    stats_path = save_root / "latent_stats.txt"

    # Plot poisoned first so clean markers are visible on top when they overlap.
    plt.figure(figsize=(8, 6))
    plt.scatter(
        proj_bd[:, 0],
        proj_bd[:, 1],
        c="#d62728",  # solid red
        s=24,
        alpha=0.8,
        label="poisoned",
        marker="o",
        linewidths=0,
        zorder=1,
    )
    plt.scatter(
        proj_clean[:, 0],
        proj_clean[:, 1],
        c="#1f77b4",  # blue
        s=28,
        alpha=0.9,
        label="clean",
        edgecolors="white",
        linewidths=0.4,
        marker="o",
        zorder=2,
    )
    plt.title(f"Latent separability ({method.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    # Basic stats
    clean_mean = clean_feats.mean(dim=0)
    bd_mean = bd_feats.mean(dim=0)
    mean_l2 = torch.norm(bd_mean - clean_mean, p=2).item()
    clean_disp = torch.norm(clean_feats - clean_mean, dim=1).mean().item()
    bd_disp = torch.norm(bd_feats - clean_mean, dim=1).mean().item()

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"method: {method}\n")
        f.write(f"total_clean_seen: {total_clean_seen}\n")
        f.write(f"total_poison_seen: {total_bd_seen}\n")
        f.write(f"plotted_clean: {clean_feats.shape[0]}\n")
        f.write(f"plotted_poison: {bd_feats.shape[0]}\n")
        f.write(f"mean_l2(clean_mean, bd_mean): {mean_l2:.6f}\n")
        f.write(f"clean_dispersion_from_clean_mean: {clean_disp:.6f}\n")
        f.write(f"poison_dispersion_from_clean_mean: {bd_disp:.6f}\n")

    return {
        "fig_path": fig_path,
        "stats_path": stats_path,
        "mean_l2": mean_l2,
        "clean_disp": clean_disp,
        "bd_disp": bd_disp,
    }
