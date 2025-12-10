import torch
import torch.nn as nn
from models.PatchTST import Model as PatchTST

class PatchTST_Classifier(nn.Module):
    """
    Wrapper around the full PatchTST forecasting model to expose only
    the classification functionality with a simplified forward() API.
    ## WHY NEEDED? no other model use this structure, redundant wrapper.
    """

    def __init__(self, configs):
        super().__init__()

        # PatchTST full model
        self.backbone = PatchTST(configs)

    def forward(self, x, padding_mask=None, x_dec=None, x_mark_dec=None, visualize=None):
        # PatchTST expects x_mark_enc, but for classification we ignore it
        x_mark_enc = None

        logits = self.backbone.classification(x, x_mark_enc)

        if visualize is not None:
            return logits, logits  # logits used as latent placeholder

        return logits
