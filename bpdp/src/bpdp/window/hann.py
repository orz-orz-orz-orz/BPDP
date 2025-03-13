"""
the hann window function
"""

import math

import torch
from torch import Tensor


def hann(t: Tensor, wl: float = 1, dtype: torch.dtype = torch.float32) -> Tensor:
    r"""
    Calculate the hann window function.
    The period is \(\pi\).
    .. math::
        h(t) = w(t) = \frac{1}{2}(1 + cos(2t) )

    Parameters
    ----------
    t: Tensor
        the timeline.
    wl: float
        the window length.
    dtype: torch.dtype
        the data type of the output tensor.

    Returns
    -------
    h: Tensor
        the output tensor

    """
    tau = t.to(torch.float64) / wl
    h = (0.5 + 0.5 * torch.cos(2 * math.pi * tau))
    h = h.masked_fill(tau.abs() > 0.5, 0.0)
    return h.to(dtype)


def t1_hann(t: Tensor, wl: float = 1, dtype: torch.dtype = torch.float32) -> Tensor:
    r"""
    calculate the hann window function products the time.
    The period is \(\pi\).
    .. math::
        h(t) = t w(t)

    Parameters
    ----------
    t: Tensor
        the timeline.
    wl: float
        the window length.
    dtype: torch.dtype
        the data type of the output tensor.

    Returns
    -------
    h: Tensor
        the output tensor

    """
    return (t.to(torch.float64) * hann(t, wl, dtype=torch.float64)).to(dtype)


def t2_hann(t: Tensor, wl: float = 1, dtype: torch.dtype = torch.float32) -> Tensor:
    r"""
    calculate the hann window function products the square of time.
    The period is \(\pi\).

    .. math::
        h(t) = t^2 w(t)

    Parameters
    ----------
    t: Tensor
        the timeline.
    wl: float
        the window length.
    dtype: torch.dtype
        the data type of the output tensor.

    Returns
    -------
    h: Tensor
        the output tensor

    """

    return (t.to(torch.float64).pow(2) * hann(t, wl)).to(dtype)


def d1_hann(t: Tensor, wl: float = 1, dtype: torch.dtype = torch.float32) -> Tensor:
    r"""
    calculate the first derivative of hann window function.
    The period is \(\pi\).

    .. math::
        w(t) = \frac{\partial }{\partial t} h(t) = h'(t) = -sin(2t)

    Parameters
    ----------
    t: Tensor
        the timeline.
    wl: float
        the window length.
    dtype: torch.dtype
        the data type of the output tensor.

    Returns
    -------
    h: Tensor
        the output tensor

    """

    tau = t.to(torch.float64) / wl
    h = - (2 * math.pi * 0.5 * torch.sin(2 * math.pi * tau))
    h = h.masked_fill(tau.abs() > 0.5, 0.0)
    return h.to(dtype)


def t1d1_hann(t: Tensor, wl: float = 1, dtype: torch.dtype = torch.float32) -> Tensor:
    r"""
    calculate the time products the first derivative of hann window function
    The period is \(\pi\).

    .. math::
        h(t) = t h'(t) = -t\sin(2t)

    Parameters
    ----------
    t: Tensor
        the timeline.
    wl: float
        the window length.
    dtype: torch.dtype
        the data type of the output tensor.

    Returns
    -------
    h: Tensor
        the output tensor

    """
    return (t.to(torch.float64) * d1_hann(t, wl, dtype=torch.float64)).to(dtype)


def t2d1_hann(t: Tensor, wl: float = 1, dtype: torch.dtype = torch.float32) -> Tensor:
    r"""
    calculate the time products the first derivative of hann window function
    The period is \(\pi\).

    .. math::
        h(t) = t^2 h'(t) = -t\sin(2t)

    Parameters
    ----------
    t: Tensor
        the timeline.
    wl: float
        the window length.
    dtype: torch.dtype
        the data type of the output tensor.

    Returns
    -------
    h: Tensor
        the output tensor

    """
    return (t.to(torch.float64).pow(2) * d1_hann(t, wl, dtype=torch.float64)).to(dtype)


def d1t1_hann(t: Tensor, wl: float = 1, dtype: torch.dtype = torch.float32) -> Tensor:
    r"""
    calculate the time products the first derivative of hann window function
    The period is \(\pi\).

    .. math::
        h(t) = (t h(t))'

    Parameters
    ----------
    t: Tensor
        the timeline.
    wl: float
        the window length.
    dtype: torch.dtype
        the data type of the output tensor.

    Returns
    -------
    h: Tensor
        the output tensor

    """
    return (t.to(torch.float64) * d1_hann(t, wl) + hann(t, wl, dtype=torch.float64)).to(dtype)
