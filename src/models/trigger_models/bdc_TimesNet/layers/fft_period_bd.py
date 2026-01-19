import torch
import torch.fft
"""
FFT Transformation function
"""
def FFT_for_Period_BD(x, k=2):
    # xf shape [B, T, C], denoting the amplitude of frequency(T) given the datapiece at B,N
    xf = torch.fft.rfft(x, dim=1)

    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)

    # Returns a new Tensor 'top_list', detached from the current graph.
    top_list = top_list.detach().cpu().numpy()
    # period: a list of shape [top_k], recording the periods of mean frequencies
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

