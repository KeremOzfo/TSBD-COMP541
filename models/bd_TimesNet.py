import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

"""
FFT Transformation function
"""
def FFT_for_Period(x, k=2):
    # xf shape [B, T, C], denoting the amplitude of frequency(T) given the datapiece at B,N
    xf = torch.fft.rfft(x, dim=1)

    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)

    # Returns a new Tensor 'top_list', detached from the current graph.
    # The result will never require gradient.Convert to a numpy instance
    top_list = top_list.detach().cpu().numpy()
    # period:a list of shape [top_k], recording the periods of mean frequencies respectively
    period = x.shape[1] // top_list
    # Here,the 2nd item returned has a shape of [B, top_k],representing the biggest top_k amplitudes 
    # for each piece of data, with N features being averaged.
    return period, abs(xf).mean(-1)[:, top_list]

"""
TimesBlock for Backdoor Trigger Generation
"""
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len  # For trigger generation, output same length
        self.k = configs.top_k # determines how many top frequencies will be considered
        # parameter-efficient design - use backdoor model dimensions
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model_bd, configs.d_ff_bd,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff_bd, configs.d_model_bd,
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
            if (self.seq_len + self.pred_len) % period != 0: # if not divisible
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous() # [B, N, period_index, T_within_period]
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N) # [B, T, N]
            # truncating down the padded part of the output and put it to result
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        # add by weight the top_k periods' result
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    TimesNet-based Backdoor Trigger Generator
    
    This model generates dynamic triggers for backdoor attacks on time series data.
    It uses the TimesNet architecture to learn input-aware triggers that can fool
    the target classifier while maintaining the time series patterns.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len  # Generate trigger of same length as input
        self.clip_ratio = configs.clip_ratio

        # stack TimesBlock for e_layers_bd times to form the trigger generator
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers_bd)])
        
        # embedding - use backdoor model dimensions
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model_bd, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers_bd
        self.layer_norm = nn.LayerNorm(configs.d_model_bd)

        # Projection layer to map from d_model_bd back to input feature dimension
        self.projection = nn.Linear(configs.d_model_bd, configs.enc_in, bias=True)

    def trigger_gen(self, x_enc, x_mark_enc=None):
        """Generate trigger for the input time series.
        
        Args:
            x_enc: Input time series [B, T, C]
            x_mark_enc: Time features (padding mask) [B, T, C]
        
        Returns:
            Trigger of same shape as input [B, T, C]
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        
        # TimesNet blocks for trigger generation
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Project back to input dimension
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, bd_labels=None):
        """Forward pass for trigger generation.
        
        Args:
            x_enc: Input time series [B, T, C]
            x_mark_enc: Padding mask [B, T, C]
            x_dec: Not used for trigger generation
            x_mark_dec: Not used for trigger generation
            bd_labels: Target backdoor labels (optional)
        
        Returns:
            trigger: Raw trigger [B, T, C]
            trigger_clipped: Amplitude-clipped trigger [B, T, C]
        """
        # Generate trigger
        dec_out = self.trigger_gen(x_enc, x_mark_enc)
        dec_out = dec_out[:, -self.pred_len:, :]
        
        # Apply amplitude clipping to keep trigger subtle
        clipped = self.clipping_amp(x_enc, dec_out, self.clip_ratio)
        
        return dec_out, clipped  # [B, L, D]

    def clipping_amp(self, x_enc, x_gen, ratio=0.1):
        """Amplitude clipping to ensure trigger is subtle.
        
        The change in value cannot be higher than a certain fraction 
        of the signal amplitude (max-min).
        
        Args:
            x_enc: Original input [B, T, C]
            x_gen: Generated trigger [B, T, C]
            ratio: Clipping ratio (default from config)
        
        Returns:
            Clipped trigger [B, T, C]
        """
        # Calculate amplitude for each channel
        max_val, _ = torch.max(x_enc, dim=1)  # [B, C]
        min_val, _ = torch.min(x_enc, dim=1)  # [B, C]
        amp = max_val - min_val  # [B, C]
        amp = amp.unsqueeze(dim=1)  # [B, 1, C]
        
        # Clip trigger to be within ±ratio * amplitude
        x_gen_clip = torch.clamp(x_gen, min=-amp * ratio, max=amp * ratio)
        
        return x_gen_clip
