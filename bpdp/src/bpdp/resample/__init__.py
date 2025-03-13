import torch

from .akima import resample_akima
from .makima import resample_makima
from .fseries import resample_fseries


def resample(x: torch.Tensor, m: int, method="makima"):
    """"
    Using various methods to resample n-point series x to m-point series.
    """
    if method == "fseries":
        return resample_fseries(x, m)
    elif method == "akima":
        return resample_akima(x, m)
    elif method == "makima":
        return resample_makima(x, m)
    else:
        raise ValueError(f"Unknown resample method {method}")

