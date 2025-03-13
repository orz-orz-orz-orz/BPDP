import torch
from functools import lru_cache

@lru_cache
def get_iu(n:int, m: int):
    t = n * (torch.arange(1, m+1)) / m - 1
    i = t.floor().int()
    i = i.view(-1, 1) + torch.arange(-2, 4).view(1, -1)
    i = i % n
    u = t % 1
    return i, u

def resample_makima(x, m):
    """
    Resample using the modified Akima spline from MATLAB.
    
    Reference: https://www.mathworks.com/help/matlab/ref/makima.html
    """
    # new index
    n = len(x)
    i, u = get_iu(n, m)
    # get the points
    y = x[i]
    # line slopes
    r = y[:, 1:] - y[:, :-1] # -2, -1, 0, 1, 2
    p = (r[:, 1:] - r[:, :-1]).abs() + (r[:, 1:] + r[:, :-1]).abs() / 2
    # spline slopes
    d0 = (p[:, 2] + p[:, 0])
    s0 = (p[:, 2] * r[:, 1] + p[:, 0] * r[:, 2]) / d0
    s0 = torch.nan_to_num(s0)
    d1 = (p[:, 3] + p[:, 1])
    s1 = (p[:, 3] * r[:, 2] + p[:, 1] * r[:, 3]) / d1
    s1 = torch.nan_to_num(s1)
    
    # coef
    a = y[:, 2]
    b = s0
    c = 3 * r[:, 2] - 2 * s0 - s1
    d = s0 + s1 - 2 * r[:, 2]
    
    x_hat = a + b * u + c * u.pow(2) + d * u.pow(3)
    return x_hat