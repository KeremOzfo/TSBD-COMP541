import torch
import torch.fft
import torch.nn.functional as F

"""
FFT Transformation function
"""
def FFT_for_Period(x, k=2):
    # xf shape [B, T, C], denoting the amplitude of frequency(T) given the datapiece at B,N
    xf = torch.fft.rfft(x, dim=1)
    #print(x.shape, xf.shape)

    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    #print(frequency_list)
    _, top_list = torch.topk(frequency_list, k)

    # Returns a new Tensor 'top_list', detached from the current graph.
    # The result will never require gradient.Convert to a numpy instance
    top_list = top_list.detach().cpu().numpy()
    # period:a list of shape [top_k], recording the periods of mean frequencies respectively
    period = x.shape[1] // top_list
    #print(period)
    # Here,the 2nd item returned has a shape of [B, top_k],representing the biggest top_k amplitudes 
    # for each piece of data, with N features being averaged.
    return period, abs(xf).mean(-1)[:, top_list]