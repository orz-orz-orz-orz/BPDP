import torch
from functools import lru_cache

@lru_cache(maxsize=30)
def resample_matrix(n, m):
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

def resample_fseries(x: torch.Tensor, m: int):
    """
    Resample using the Fourier series.
    """
    w = resample_matrix(x.shape[0], m)
    return x @ w
