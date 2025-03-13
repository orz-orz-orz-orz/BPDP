import torch
import torch.nn.functional as F
from operator import itemgetter

from .filter import bp1, bp2
from .resample import resample

def cost_fn(x, n2, n1, n0, alpha, short=50, long=500, resampler='makima', debug=False):
    # 每次看前 2 個 pitch mark
    # n2 < n1 < n0
    # 如果是 restart 狀況，代表前面一個 片段 [n2+1, n1] 是使用 sum of squared
    x0 = x[n1 + 1: n0 + 1]
    x1 = x[n2 + 1: n1 + 1]

    if len(x0) < short:  # 不允許過短的
        if debug: print(f"({n2}, {n1}, {n0}): too short")
        return torch.inf

    if len(x0) > long or len(x1) > long: 
        if debug: print(f"({n2}, {n1}, {n0}): too long")
        return alpha[n0] - alpha[n1]

    if 2 * len(x0) < len(x1) or 2 * len(x1) < len(x0) :
        if debug: print(f"({n2}, {n1}, {n0}): too different")
        return alpha[n0] - alpha[n1]
    # resample x1 to x0
    x0_ = resample(x1, n0 - n1, method=resampler)

    a = x0 @ x0_ / max(x0_ @ x0_, 1e-8)

    if a < 0:  # 不可以是負相關
        if debug: print(f"({n2}, {n1}, {n0}): drop the negative correlation point.")
        return torch.inf

    r0 = (x0 - a * x0_).pow(2)

    if debug: print(f"({n2}, {n1}, {n0}): using the residual.")

    return r0.sum()


def bpdp(x, sr=24000, wl_0=0.05, wl_1=0.002, f_lo=50.0, f_hi=550.0, beam_size=5, filter="bp1", resampler='makima', _probe=None):
    """
    Pitch marks extraction using Band-pass filtering with Dynamic Programming

    Parameters
    ----------
    x: torch.Tensor
        The input signal of length L.
    sr: int
        The sample rate.
    wl_0: float
        The window length 0
    wl_1: float
        The window length 1
    f_lo: float
        The minimum f0 in Hz (low).
    f_hi: float
        The maximum f0 in Hz (high)
    beam_size: int
        The beam size
    filter: str
        The filtering method
    resampler: str
        The resampling method
    _probe: (int, int) | None
        Print the decision between samples (n0, n1). 

    Returns
    -------
    p: Tuple[int, ...]
        The indices of pitch marks.

    """

    # create band-pass filter
    # h = bp2(fl, sr, f_lo, f_hi)
    if filter == 'bp2':
        g, h = bp2(sr, wl_0, wl_1)
        # apply the filter 0
        y = F.conv1d(x.view(1, 1, -1), g.view(1, 1, -1), padding=g.shape[0]//2).view(-1)[:x.shape[-1]]
        y = x - y
        # apply the filter 1
        y = F.conv1d(y.view(1, 1, -1), h.view(1, 1, -1), padding=h.shape[0]//2).view(-1)[:x.shape[-1]]
    elif filter == 'bp1':
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
        # check the range
        need_probe = (_probe is not None) and (_probe[0] < s_k < _probe[1])
        # A DICT FOR NEW CREATED SUBSETS
        new_subsets = {}
        for u in range(len(subsets)):
            # GET PREVIOUS SUBSET
            cost, seq = subsets[u]
            s_j = seq[-1]
            #  CALCULATE NEW SEGMENT COST
            if len(seq) > 1:
                s_i = seq[-2]
                cost_u = cost_fn(y, s_i, s_j, s_k, alpha, short, long, resampler, debug=need_probe)
            else:
                cost_u = alpha[s_k] - alpha[s_j]
            suffix = (s_j, s_k)
            # new_cost = cost + cost_u - beta[s_j] + beta[s_k]
            new_cost = cost + ((beta[s_k] - beta[s_j]) + cost_u)
            # NEW COST-SUBSET PAIR
            new_subset = (new_cost, (*seq, s_k))

            if need_probe:
                print(f"new_node = ({new_subset[0]}, {new_subset[1][-5:]})")

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
        
        if need_probe:
            print(f"stack after add new node {s_k}")
            for i, subset in enumerate(subsets):
                print(f"[{i}] = ({subset[0]}, {subset[1][-5:]})")

    # GET THE BEST SUBSET
    subsets.sort(key=itemgetter(0))
    p = subsets[0][1]

    return p