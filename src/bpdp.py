import math
import torch
import torchaudio
import torch.nn.functional as F

from .window.k6 import k6
from functools import lru_cache
from operator import itemgetter


def lp(t, fc, sr):
    c = torch.sinc(2 * fc * t ) * 2 * fc / sr
    return c

@lru_cache(maxsize=3)
def bp(fl:int, sr: int, f_lo: float, f_hi: float) -> torch.Tensor:
    """
    Create a symmetric band-pass filter with sinc-filter and cosine modulation.

    Parameters
    ----------
    fl: int
        Filter length
    sr: int
        The sample rate.
    f_lo: float
        Cut-off frequency (Low). Unit is Hz.
    f_hi: float
        Cut-off frequency (High). Unit is Hz
    

    Returns
    -------
    h: torch.Tensor
        The result pitch marks length L

    """
    # create band-pass filter
    t = (torch.arange(fl) - fl //2) / sr
    g0 = k6(t, wl=fl / sr) * lp(t, f_lo, sr)
    g0 /= g0.sum()
    g1 = k6(t, wl=fl / sr * f_lo / f_hi) * lp(t, f_hi, sr)
    g1 /= g1.sum()

    h = g1 - g0
    return h

@lru_cache(maxsize=3)
def bp1(sr: int, wl_0: float, wl_1: float, f_hi=500.0) -> torch.Tensor:
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
def bp2(sr, wl_0=0.1, wl_1=0.0125) -> torch.Tensor:
    """
    Create a symmetric band-pass filter with sinc-filter and cosine modulation.

    Parameters
    ----------
    fl: int
        Filter length
    sr: int
        The sample rate.
    f_lo: float
        Cut-off frequency (Low). Unit is Hz.
    f_hi: float
        Cut-off frequency (High). Unit is Hz
    

    Returns
    -------
    h: torch.Tensor
        The result pitch marks length L

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



@lru_cache(maxsize=30)
def dft_matrix(n, m):
    PI = torch.pi
    k = torch.arange(n)
    u = torch.arange(1, n + 1)  # [1, n]
    v = torch.arange(1, m + 1) * n / m  # [1, m]
    # DFT
    cos = torch.cos(2 * PI * torch.outer(u, k) / n)  # (n, n)
    sin = torch.sin(2 * PI * torch.outer(u, k) / n)  # (n, n)
    a = torch.cat((cos, sin), dim=1)  # (n, 2n)

    # interpolation
    cos = torch.cos(2 * PI * torch.outer(k, v) / n) / n  # (n, m)
    sin = torch.sin(2 * PI * torch.outer(k, v) / n) / n  # (n, m)

    if n > m: # anti-aliasing
        cos[m:, :] = 0.0
        sin[m:, :] = 0.0

    b = torch.cat((cos, sin), dim=0)  # (2n, m)
    return a @ b


def cost_fn(x, n2, n1, n0, short=50, long=500, debug=False):
    # 每次看前 2 個 pitch mark
    # n2 < n1 < n0
    # 如果是 restart 狀況，代表前面一個 片段 [n2+1, n1] 是使用 sum of squared
    x0 = x[n1 + 1: n0 + 1]
    x1 = x[n2 + 1: n1 + 1]

    if len(x0) < short:  # 不允許過短的
        if debug: print("too short")
        return torch.inf

    p0 = x0.pow(2)

    if len(x0) > long or len(x1) > long: 
        if debug: print("current frame or previous frame is too long.")
        return p0.sum()

    if 2 * len(x0) < len(x1) or 2 * len(x1) < len(x0) :
        if debug: print("the lengths of two frames are too different.")
        return p0.sum() 
    # resample x1 to x0
    w = dft_matrix(n1 - n2, n0 - n1)
    x0_ = x1 @ w

    #m0 = x0.mean()
    #m1 = x0_.mean()
    #N = len(x0)
    # a = ((x0 @ x0_) - N / 2 * m0 * m1) / max((x0_ @ x0_) - N / 2 * m1 * m1, 1e-5)
    a = x0 @ x0_ / max(x0_ @ x0_, 1e-8)

    if a < 0:  # 不可以是負相關
        if debug: print("drop the negative correlation point.")
        return torch.inf

    r0 = (x0 - a * x0_).pow(2)

    #r = torch.minimum(p0, r0)

    if debug: print("using the residual.")
    return r0.sum()


def bpdp(x, sr=24000, wl_0=0.05, wl_1=0.002, f_lo=50.0, f_hi=550.0, beam_size=5, filt="bp1"):
    """
    Pitch marks extraction using Band-pass filtering with Dynamic Programming

    Parameters
    ----------
    x: torch.Tensor
        The input signal of length L.
    sr: int
        The sample rate.
    fl: int
        The frame length (= Length of the band-pass filter).
    f_lo: float
        The cut-off frequency (low).
    f_hi: float
        The cut-off frequency (high)

    Returns
    -------
    p: torch.Tensor
        The pitch marks.

    """

    # create band-pass filter
    # h = bp2(fl, sr, f_lo, f_hi)
    if filt == 'bp2':
        g, h = bp2(sr, wl_0, wl_1)
        # apply the filter 0
        y = F.conv1d(x.view(1, 1, -1), g.view(1, 1, -1), padding=g.shape[0]//2).view(-1)[:x.shape[-1]]
        y = x - y
        # apply the filter 1
        y = F.conv1d(y.view(1, 1, -1), h.view(1, 1, -1), padding=h.shape[0]//2).view(-1)[:x.shape[-1]]
    elif filt == 'bp1':
        h = bp1(sr, wl_0, wl_1)
        y = F.conv1d(x.view(1, 1, -1), h.view(1, 1, -1), padding=h.shape[0]//2).view(-1)[:x.shape[-1]]


    # get the zero crossing points
    z = torch.zeros_like(y)
    z[1:] += (y[:-1] < 0) & (y[1:] >= 0)


    # short and long
    short = int(sr / f_hi)
    long = int(sr / f_lo + 0.5)

    # create the candidates
    c_list = []
    for i in torch.nonzero(z).view(-1):
        c_list.append(int(i))

    # forward error and backward error
    alpha = torch.cumsum(y.pow(2), dim=0)
    beta = alpha[-1] - alpha

    subsets = []
    for k in range(len(c_list)):
        # GET A NEW POINT
        s_k = c_list[k]
        # A DICT FOR NEW CREATED SUBSETS
        new_subsets = {}
        for u in range(len(subsets)):
            # GET PREVIOUS SUBSET
            cost, seq = subsets[u]
            s_j = seq[-1]
            #  CALCULATE NEW SEGMENT COST
            if len(seq) > 1:
                s_i = seq[-2]
                cost_u = cost_fn(y, s_i, s_j, s_k, short, long)
            else:
                cost_u = alpha[s_k] - alpha[s_j]
            suffix = (s_j, s_k)
            new_cost = cost + cost_u - beta[s_j] + beta[s_k]
            # NEW COST-SUBSET PAIR
            new_subset = (new_cost, (*seq, s_k))
            # MERGE THE SUBSETS WITH SAME SUFFIX
            if suffix in new_subsets:
                min_subset = new_subsets[suffix]
                if new_cost < min_subset[0]:
                    new_subsets[suffix] = new_subset
            else:
                new_subsets[suffix] = new_subset
        new_subsets = list(new_subsets.values())
        # ADD A ONE-ITEM SUBSET AS START POINT
        if k < beam_size:
            new_subsets.append((alpha[-1], (s_k,)))
        # ADD NEW SUBS TO LIST OOOF SUBSETS
        subsets = new_subsets + subsets
        # SHRINK THE LIST OF SUBSETS
        if len(subsets) > beam_size:
            subsets.sort(key=itemgetter(0))
            subsets = subsets[:beam_size]
    # GET THE BEST SUBSET
    subsets.sort(key=itemgetter(0))
    p = subsets[0][1]

    return p