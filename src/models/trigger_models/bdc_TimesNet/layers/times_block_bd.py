import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Conv_Blocks import Inception_Block_V1

from .fft_period_bd import FFT_for_Period_BD

"""
Conditional TimesBlock for Backdoor Trigger Generation
"""
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len  # For trigger generation, output same length
        self.k = configs.top_k  # determines how many top frequencies will be considered
        self.d_model_bd = configs.d_model_bd
        
        # parameter-efficient design - use backdoor model dimensions
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model_bd, configs.d_ff_bd,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff_bd, configs.d_model_bd,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x, label_cond=None):
        """
        Forward pass with optional label conditioning
        
        Args:
            x: B x T x d_model_bd
            label_cond: B x 1 x d_model_bd (optional conditioning signal)
        """
        B, T, N = x.size()

        # period_list([topk]) -> topk significant frequency
        # period_weight([B, topk]) -> their amplitudes 
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding - use actual T instead of seq_len + pred_len
            if T % period != 0:
                length = ((T // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - T), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = T
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            # truncating down the padded part
            res.append(out[:, :T, :])
        res = torch.stack(res, dim=-1)
        
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        
        # Add label conditioning if provided
        if label_cond is not None:
            res = res + label_cond
        
        # residual connection
        res = res + x
        return res

