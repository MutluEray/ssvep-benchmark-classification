"""Signal pre-processing steps shared by both pipelines.

All functions operate on arrays whose LAST axis is time.
"""
import numpy as np
from scipy.signal import butter, filtfilt


def demean(x):
    """Remove the mean of each trial/channel (as in the original loader of pipeline B)."""
    return x - x.mean(axis=-1, keepdims=True)


def butter_bandpass(x, fs, low, high, order=4):
    """Zero-phase Butterworth band-pass (pipeline A, FBCCA sub-bands)."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x, axis=-1)


def fir_bandpass(x, fs, low, high):
    """MNE default zero-phase FIR band-pass (pipeline B, as in the original notebook)."""
    import mne
    return mne.filter.filter_data(x, sfreq=fs, l_freq=low, h_freq=high, verbose=False)


def clip(x, fs, window_s):
    """Keep samples between window_s[0] and window_s[1] seconds of the epoch."""
    start, stop = int(window_s[0] * fs), int(window_s[1] * fs)
    return x[..., start:stop]


def zscore_across_trials(x):
    """Z-score each (channel, time point) across all trials, as in the original pipeline B.

    Note: statistics are computed on ALL trials (train and test). This is unsupervised and
    reproduces the original; it is a mild form of transductive normalisation.
    """
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def combine_channels(x, method="weighted", weights=None):
    """Collapse the channel axis of (n_trials, n_channels, n_times) to (n_trials, n_times)."""
    if method == "mean":
        return x.mean(axis=1)
    if method == "median":
        return np.median(x, axis=1)
    if method == "weighted":
        w = np.asarray(weights, dtype=float)
        return np.average(x, axis=1, weights=w / w.sum())
    raise ValueError(f"Unknown channel combination method: {method}")
