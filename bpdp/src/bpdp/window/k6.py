"""
Kawahara's new 6-terms cosine series
Source: http://arxiv.org/abs/1702.06724
"""

import math

import torch
from torch import Tensor


def k6(t: Tensor, wl: float = 1, dtype: torch.dtype = torch.float32):
    r"""
    Calculate Kawahara's 6-terms cosine series at time t.
    The window's length is equal to argument length.
    .. math::
        h(t) = w(t) = \sum_{k=0}^{5} a_k \cos(k 2 \pi \frac{t}{wl})

    The frequency response at zero frequency \[\hat h (0) = a_0 \cdot L \]
    The -3db Point is about 1.289 / L.

    Parameters
    ----
    t: Tensor
        the timeline
    wl: Tensor
        the length of the window
    dtype: torch.dtype
        the data type of the result tensor

    Returns
    -------
    w : Tensor
        the output tensor
    """
    tau = (t.to(torch.float64) / wl)

    a = [0.2624710164, 0.4265335164, 0.2250165621, 0.0726831633, 0.0125124215, 0.0007833203]
    w = a[0] \
        + a[1] * torch.cos(2 * math.pi * tau) \
        + a[2] * torch.cos(4 * math.pi * tau) \
        + a[3] * torch.cos(6 * math.pi * tau) \
        + a[4] * torch.cos(8 * math.pi * tau) \
        + a[5] * torch.cos(10 * math.pi * tau)
    w = w.masked_fill(tau.abs() > 0.5, 0.0)

    return w.to(dtype)


def t1_k6(t: Tensor, wl: float = 1, dtype: torch.dtype = torch.float32):
    r"""
    Calculate Kawahara's 6-terms cosine series times t

    Args
    ----
    t: Tensor
        the timeline
    wl: Tensor
        the length of the window
    dtype: torch.dtype
        the data type of the result tensor


    Returns
    -------
    w : Tensor
        the output tensor
    """
    return (t.to(torch.float64) * k6(t, wl, dtype=torch.float64)).to(dtype)


def t2_k6(t: Tensor, wl: float = 1, dtype: torch.dtype = torch.float32):
    r"""
    Calculate Kawahara's 6-terms cosine series times t^2

    Args
    ----
    t: Tensor
        the timeline
    wl: Tensor
        the length of the window
    dtype: torch.dtype
        the data type of the result tensor

    Returns
    -------
    w : Tensor
        the output tensor
    """
    return (t.to(torch.float64).pow(2) * k6(t, wl, dtype=torch.float64)).to(dtype)


def d1_k6(t: Tensor, wl: float = 1, dtype: torch.dtype = torch.float32):
    r"""
    Calculate the time derivative of Kawahara's 6-terms cosine series

    Args
    ----
    t: Tensor
        the timeline
    wl: Tensor
        the length of the window
    dtype: torch.dtype
        the data type of the result tensor

    Returns
    -------
    w : Tensor
        the output tensor
    """
    tau = (t.to(torch.float64) / wl)
    a = [0.2624710164, 0.4265335164, 0.2250165621, 0.0726831633, 0.0125124215, 0.0007833203]
    w = -2 * math.pi * (a[1] * torch.sin(2 * math.pi * tau)
                        + 2 * a[2] * torch.sin(4 * math.pi * tau)
                        + 3 * a[3] * torch.sin(6 * math.pi * tau)
                        + 4 * a[4] * torch.sin(8 * math.pi * tau)
                        + 5 * a[5] * torch.sin(10 * math.pi * tau)) / wl
    w = w.masked_fill(tau.abs() > 0.5, 0.0)
    return w.to(dtype)


def t1d1_k6(t: Tensor, wl: float = 1, dtype: torch.dtype = torch.float32):
    r"""
    Calculate the time derivative of Kawahara's 6-terms cosine series

    Args
    ----
    t: Tensor
        the timeline
    wl: Tensor
        the length of the window
    dtype: torch.dtype
        the data type of the result tensor

    Returns
    -------
    w : Tensor
        the output tensor
    """
    return (t.to(torch.float64) * d1_k6(t, wl, dtype=torch.float64)).to(dtype)


def t2d1_k6(t: Tensor, wl: float = 1, dtype: torch.dtype = torch.float32):
    r"""
    Calculate the time derivative of Kawahara's 6-terms cosine series

    Args
    ----
    t: Tensor
        the timeline
    wl: Tensor
        the length of the window
    dtype: torch.dtype
        the data type of the result tensor

    Returns
    -------
    w : Tensor
        the output tensor
    """
    return (t.to(torch.float64).pow(2) * d1_k6(t, wl, dtype=torch.float64)).to(dtype)


def d1t1_k6(t: Tensor, wl: float = 1, dtype: torch.dtype = torch.float32):
    r"""
    Calculate the time derivative of Kawahara's 6-terms cosine series
    \[
        DTh(t) = (th)' (t)
    \]
    Args
    ----
    t: Tensor
        the timeline
    wl: Tensor
        the length of the window
    dtype: torch.dtype
        the data type of the result tensor

    Returns
    -------
    w : Tensor
        the output tensor
    """
    return (k6(t, wl, dtype=torch.float64) + t1d1_k6(t, wl, dtype=torch.float64)).to(dtype)

# class _CallableModule(ModuleType):
#    def __call__(self, t: Tensor, wl: float, dtype: torch.dtype = torch.float32) -> Tensor:
#        return k6(t, wl, dtype)


# sys.modules[__name__].__class__ = _CallableModule
