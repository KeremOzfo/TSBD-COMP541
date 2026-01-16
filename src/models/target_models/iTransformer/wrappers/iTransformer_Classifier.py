import torch
import torch.nn as nn
from ..model import Model as ITransformer


class iTransformer_Classifier(nn.Module):
    """
    Wrapper around the full iTransformer model to expose only
    the classification functionality with a simplified forward() API.
    """

    def __init__(self, configs):
        super().__init__()

        # Full iTransformer backbone
        self.backbone = iTransformer(configs)

    def forward(
        self,
        x,
        padding_mask=None,
        x_dec=None,
        x_mark_dec=None,
        visualize=None
    ):
        # iTransformer classification ignores x_mark_enc
        x_mark_enc = None

        logits = self.backbone.classification(x, x_mark_enc)

        if visualize is not None:
            # logits used as latent placeholder
            return logits, logits

        return logits
