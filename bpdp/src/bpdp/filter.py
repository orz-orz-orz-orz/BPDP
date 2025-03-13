import torch
from .window.k6 import k6
from functools import lru_cache
from typing import Tuple

@lru_cache(maxsize=3)
def bp1(sr: int, wl_0: float, wl_1: float) -> torch.Tensor:
    r"""
    Create a symmetric using two windows of lengths wl_0 and wl_1.
    The window length \(L = \lfloor wl_1 * sr + 1 \rfloor \). 

    Parameters
    ----------
    sr: int
        The sample rate in Hz.
    wl_0: float
        Window length 0 in msec.
    wl_1: float
        Window length 1 in msec.
    

    Returns
    -------
    h, g: (torch.Tensor, torch.Tensor)
        Two windows.
    """
    # create band-pass filter
    fl = wl_0 * sr
    t = (torch.arange(fl) - fl //2) / sr
    g0 = k6(t, wl=wl_0)
    g0 /= g0.sum()
    g1 = k6(t, wl=wl_1)
    g1 /= g1.sum()

    h = g1 - g0
    return h


@lru_cache(maxsize=3)
def bp2(sr, wl_0=0.1, wl_1=0.0125) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create two-low pass filters with wl_0 and wl_1.
    The window length in sample L = int( wl * sr + 1).

    Parameters
    ----------
    sr: int
        The sample rate in Hz.
    wl_0: float
        Window length 0 in msec.
    wl_1: float
        Window length 1 in msec.
    

    Returns
    -------
    (w0, w1): (torch.Tensor, torch.Tensor)
        windows of lengths wl_0 and wl_1.
    """
    # create filter 0
    fl = int(wl_0 * sr + 1)
    t = (torch.arange(fl) - fl //2) / sr
    g = k6(t, wl=wl_0)
    g /= g.sum()

    # create filter 1
    fl = int(wl_1 * sr + 1)
    t = (torch.arange(fl) - fl //2) / sr
    h = k6(t, wl=wl_1)  # 0.0125 => 接近 ZFF (p=400)
    h /= h.sum()
    return g, h