import torch
import torch.nn as nn
import torch.nn.functional as F

from .fft_period import FFT_for_Period
from layers.Conv_Blocks import Inception_Block_V1
"""
TimesBlock
Gets base frequencies of the data by applying FFT transformation,
and then reshapes the data to 2D respecting these frequences.
"""
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k # determines how many top frequencies will be considered
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size() # B: batch size, T: sequence length, N: feature number

        # period_list([topk]) -> topk significant frequency
        # period_weight([B, topk]) -> their amplitudes 
        period_list, period_weight = FFT_for_Period(x, self.k)


        res = []
        for i in range(self.k): # for each top_k frequency
            period = period_list[i] # get the period
            # padding
            # total length of the sequence + length of the part that will be predicted
            # needs to be divisible by the period, so it adds padding to handle this.
            if (self.seq_len + self.pred_len) % period != 0: # if not divisible
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            # we need each channel of a single piece of data to be a 2D variable,
            # Also, in order to implement the 2D conv later on, we need to adjust the 2 dimensions 
            # to be convolutioned to the last 2 dimensions, by calling the permute() func.
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous() # [B, N, period_index, T_within_period]
            #print(out.shape,'outt')
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            #print(out.shape, 'outttt')
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N) # [B, T, N]
            # truncating down the padded part of the output and put it to result
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        # First, use softmax to get the normalized weight from amplitudes --> 2D [B,top_k]
        period_weight = F.softmax(period_weight, dim=1)
        # after two unsqueeze(1),shape -> [B,1,1,top_k],so repeat the weight to fit the shape of res
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        # add by weight the top_k periods' result, getting the result of this TimesBlock
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res
