# BPDP: A PyTorch implementation of our BPDP epoch extraction algorithm.
## Description
This package implementation the BPDP epoch extraction algorithm using Pytorch. 

## Install
This package can be installed using pip.
```
pip install bpdp
```

## Example Usage:
```Python
import bpdp
import torchaudio

x, sr = torchaudio.load("example.wav")
epochs = bpdp.bpdp(x, sr)
```

## Documentation
This package only contains a few functions so we just write the doc here.

### def bpdp.bpdp(x, sr, wl_0=0.05, wl_1=0.002, f_lo=50.0, f_hi=550.0, beam_size=5, filter="bp1", resampler='makima', _probe=None) -> Tuple[int, ...]:
The main entry point of our bpdp algorithm.

#### Parameters:
- x(torch.Tensor): The input signal of shape(L).
- sr(int): The sample rate in Hz.
- wl_0(float): The window length of the high pass filter in ms (Default=0.05).
- wl_1(float): The window length of the low pass filter in ms (Default=0.002).
- f_lo(float): The minimum fundamental frequency in Hz (Default=50.0).
- f_hi(float): The maximum fundatental frequency in Hz (Default=550.0).
- beam_size(int): The beam size used in dynamic programming (Default=5).
- filter(str): The filtering method. Should be one of:
    - 'bp1': Apply one bandpass filter (Default).
    - 'bp2': Apply a highpass filter and a lowpass filter sequentially.
- resampler(str): The resample method. Should be one of:
    - 'akima': Akima piecewise cubic Hermite interpolation.
    - 'makima': Modified Akima piecewise cubic Hermite interpolation (Default).
    - 'fseries': Inteplation using the Fourier series.
- _probe(Tuple[int, int] or None): Print the decision between sample n0 to n1. The input should be a 2-tuple containing n0 and n1 (Default=None).

#### Returns:
- p(Tuple[int, ...]): The indices of epoch in sample.

### def bpdp.filter.bp1(sr, wl_0, wl_1) -> torch.Tensor:
Return a bandpass filter using K6 window.
#### Parameters:
- sr(int): The sample rate $f_s$ in Hz.
- wl_0(float): The window length $T_0$ in msec.
- wl_1(float): The window length $T_1$ in msec.
#### Return:
- h(torch.Tensor): A bandpass filter of size $L=\lfloor T_0  f_s + 1\rfloor$

### def bpdp.filter.bp2(sr, wl_0, wl_1) -> Tuple[torch.Tensor, torch.Tensor]:
Return a highpass filter and a lowpass filter using K6 window.
#### Parameters:
- sr(int): The sample rate $f_s$ in Hz.
- wl_0(float): The window length $T_0$ in msec.
- wl_1(float): The window length $T_1$ in msec.
#### Return:
- h(torch.Tensor): A highpass filter of size $L=\lfloor T_0  f_s + 1\rfloor$
- g(torch.Tensor): A lowpass filter of size $L=\lfloor T_1  f_s + 1\rfloor$
### def bpdp.resample.resample(x, m, method='makima'):
Resample the $N$-periodic signal $x$ to $M$-periodic signal $y$.
#### Parameters:
- x(torch.Tensor): The input vector $x$ of size $N$.
- m(int): The output period $M$.
- method(str): The interpolation method. Should be one of:
    - 'akima': Akima piecewise cubic Hermite interpolation.
    - 'makima': Modified Akima piecewise cubic Hermite interpolation (Default).
    - 'fseries': Inteplation using the Fourier series.
#### Returns
- y(torch.Tensor): The output vector $y$ of size $M$.

### def bpdp.resample.resample_akima(x, m):
Resample the $N$-periodic signal $x$ to $M$-periodic signal $y$ using the Akima spline interpolation.
#### Parameters:
- x(torch.Tensor): The input vector $x$ of size $N$.
- m(int): The output period $M$.
#### Returns
- y(torch.Tensor): The output vector $y$ of size $M$.

### def bpdp.resample.resample_makima(x, m):
Resample the $N$-periodic signal $x$ to $M$-periodic signal $y$ using the modified Akima spline interpolation.
#### Parameters:
- x(torch.Tensor): The input vector $x$ of size $N$.
- m(int): The output period $M$.
#### Returns
- y(torch.Tensor): The output vector $y$ of size $M$.

### def bpdp.resample.resample_fseries(x, m):
Resample the $N$-periodic signal $x$ to $M$-periodic signal $y$ using the Fourier series.
#### Parameters:
- x(torch.Tensor): The input vector $x$ of size $N$.
- m(int): The output period $M$.
#### Returns
- y(torch.Tensor): The output vector $y$ of size $M$.

### def bpdp.window.k6.k6(t, wl = 1.0, dtype=torch.float32):
Calculate Kawahara's 6-terms cosine series at time $t$.
The window's length $L$ is equal to `wl`.
$$
        h(t) = w(t) = \sum_{k=0}^{5} a_k \cos(k 2 \pi \frac{t}{L})
$$
The frequency response at zero frequency $\hat h (0) = a_0  L.$ 
The -3db Point is about $1.289 / L$.
#### Parameters:
- t(torch.Tensor): The time $t$ for calculation, can be in any shape.
- wl(float): The window length.
- dtype(torch.dtype):  The data type of the output.
#### Returns:
- h(torch.Tensor): The output tensor. Its shape is same as input tensor $t$