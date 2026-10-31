"""Training-free SSVEP detectors used as reference points.

These were NOT part of the original course project. They are standard methods that any
SSVEP classifier should be compared against:
  * PSDA rule  - pick the stimulus frequency nearest to the spectral peak in 7.9-15.9 Hz
  * CCA        - Lin et al. (2006): canonical correlation with sin/cos harmonic references
  * FBCCA      - Chen et al. (2015): CCA over a bank of harmonic sub-bands
"""
import numpy as np

import config
from preprocessing import clip


# ----------------------------------------------------------------------------- PSDA rule
def nearest_target(freq, stim_freqs, valid_range=config.PSDA_RANGE):
    """Map frequencies to the nearest stimulus target; -1 if outside the stimulus range."""
    freq = np.asarray(freq, dtype=float)
    tgt = np.argmin(np.abs(freq[..., None] - stim_freqs[None, :]), axis=-1)
    outside = (freq < valid_range[0]) | (freq > valid_range[1])
    return np.where(outside, -1, tgt)


def psda_single_trial(wave, fs, stim_freqs, valid_range=config.PSDA_RANGE):
    """Single-trial PSDA on a channel-combined waveform (n_trials, n_times)."""
    n = wave.shape[-1]
    spec = np.abs(np.fft.rfft(wave, axis=-1))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    idx = np.where((freqs >= valid_range[0]) & (freqs <= valid_range[1]))[0]
    peak = freqs[idx[np.argmax(spec[:, idx], axis=-1)]]
    return nearest_target(peak, stim_freqs, valid_range)


def psda_block_vote(block_peaks, stim_freqs, valid_range=config.PSDA_RANGE):
    """Majority vote of nearest targets over the 6 block peak frequencies (pipeline A samples)."""
    votes = nearest_target(block_peaks, stim_freqs, valid_range)   # (n, 6)
    out = np.full(votes.shape[0], -1)
    for i, v in enumerate(votes):
        v = v[v >= 0]
        if v.size:
            out[i] = np.bincount(v).argmax()
    return out


# ----------------------------------------------------------------------------- CCA
def harmonic_references(stim_freqs, n_samples, fs, n_harmonics):
    """Orthonormal bases of sin/cos references: (n_targets, n_samples, 2 * n_harmonics)."""
    t = np.arange(n_samples) / fs
    refs = []
    for f in stim_freqs:
        cols = []
        for h in range(1, n_harmonics + 1):
            cols += [np.sin(2 * np.pi * h * f * t), np.cos(2 * np.pi * h * f * t)]
        y = np.stack(cols, axis=1)
        y -= y.mean(axis=0)
        refs.append(np.linalg.qr(y)[0])
    return np.stack(refs)


def cca_correlations(x, q_refs):
    """Largest canonical correlation between each trial and each reference.

    x: (n_trials, n_channels, n_times); q_refs: (n_targets, n_times, 2*n_harmonics)
    returns (n_trials, n_targets)
    """
    xt = np.transpose(x, (0, 2, 1))
    xt = xt - xt.mean(axis=1, keepdims=True)
    qx = np.linalg.qr(xt)[0]                                    # (n, T, C)
    m = np.einsum("ntc,ktr->nkcr", qx, q_refs, optimize=True)   # (n, K, C, R)
    return np.linalg.svd(m, compute_uv=False)[..., 0]


def cca_predict(x, stim_freqs, fs, n_harmonics=config.CCA["n_harmonics"]):
    q = harmonic_references(stim_freqs, x.shape[-1], fs, n_harmonics)
    return np.argmax(cca_correlations(x, q), axis=1)


def fbcca_scores(x_bands, q_refs, weights):
    """FBCCA scores from already filtered and windowed data (n_trials, n_bands, n_channels, n_times)."""
    rho = 0.0
    for b, w in enumerate(weights):
        rho = rho + w * cca_correlations(x_bands[:, b], q_refs) ** 2
    return rho


def fbcca_predict(x_full, stim_freqs, fs, window_s=config.BASELINE_WINDOW_S, cfg=config.FBCCA):
    """FBCCA (Chen et al., 2015, M3) on full demeaned epochs: filter bank first, then the window."""
    from filterbank import band_weights, filter_bank
    xb = clip(filter_bank(x_full, fs, cfg["n_subbands"]), fs, window_s)
    q = harmonic_references(stim_freqs, xb.shape[-1], fs, cfg["n_harmonics"])
    return np.argmax(fbcca_scores(xb, q, band_weights(cfg["n_subbands"])), axis=1)
