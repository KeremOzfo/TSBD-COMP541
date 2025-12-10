import torch


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        """Create an upper-triangular causal mask for attention.

        Args:
            B: Batch size.
            L: Sequence length.
            device: Torch device for the mask tensor.

        Attributes:
            _mask: Boolean tensor of shape (B, 1, L, L) where True denotes
                masked future positions (strictly upper triangle).
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        """Construct probabilistic attention mask based on provided indices.

        Args:
            B: Batch size.
            H: Number of attention heads.
            L: Sequence length.
            index: Long tensor of indices selecting query positions per batch
                and head, shape (B, H, L_q).
            scores: Attention score tensor whose last dim defines key length,
                used only for shaping the final mask.
            device: Torch device for the mask tensor.

        Attributes:
            _mask: Boolean tensor broadcastable to `scores.shape`, where True
                denotes masked positions.
        """
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
