"""Filter bank and analysis windows used by FBCCA, eCCA and ensemble TRCA.

Sub-band n (1-based) follows Chen et al. (2015, method M3) as implemented in the reference
TRCA code (Nakanishi et al., 2018): Chebyshev type I, passband [8n-2, 90] Hz, zero-phase.
"""
from functools import lru_cache

import numpy as np
from scipy.signal import cheb1ord, cheby1, sosfiltfilt

import config

PASS_LOW = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
STOP_LOW = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]


@lru_cache(maxsize=None)
def _sos(n, fs):
    nyq = fs / 2.0
    wp = [PASS_LOW[n - 1] / nyq, 90.0 / nyq]
    ws = [STOP_LOW[n - 1] / nyq, 100.0 / nyq]
    order, wn = cheb1ord(wp, ws, 3, 40)
    return cheby1(order, 0.5, wn, btype="bandpass", output="sos")


def filter_bank(x, fs, n_bands):
    """(n_trials, n_channels, n_times) -> (n_trials, n_bands, n_channels, n_times). Filter full epochs."""
    return np.stack([sosfiltfilt(_sos(n, fs), x, axis=-1) for n in range(1, n_bands + 1)], axis=1)


def band_weights(n_bands, a=config.FBCCA["a"], b=config.FBCCA["b"]):
    n = np.arange(1, n_bands + 1, dtype=float)
    return n ** (-a) + b


def window(x, fs, length_s, latency_s=config.LATENCY_S, onset_s=config.STIM_ONSET_S):
    """Analysis window of `length_s` seconds starting at stimulus onset + visual latency."""
    start = int(round((onset_s + latency_s) * fs))
    return x[..., start:start + int(round(length_s * fs))]
