"""Spectral peak-frequency features."""
import numpy as np


def _amplitude_spectrum(x, fs, nfft=None):
    n = x.shape[-1] if nfft is None else int(nfft)
    spec = np.abs(np.fft.fft(x, n=n, axis=-1))[..., : n // 2]
    freqs = np.fft.fftfreq(n, d=1.0 / fs)[: n // 2]
    return freqs, spec


def peak_frequency(x, fs, nfft=None):
    """Frequency of the largest amplitude-spectrum bin (pipeline A feature). Shape: x.shape[:-1]."""
    freqs, spec = _amplitude_spectrum(x, fs, nfft)
    return freqs[np.argmax(spec, axis=-1)]


def band_peak_frequencies(x, fs, bands, nfft=None):
    """Peak frequency within each (low, high) band (pipeline B features). Shape: x.shape[:-1] + (n_bands,)."""
    freqs, spec = _amplitude_spectrum(x, fs, nfft)
    out = []
    for low, high in bands:
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        if idx.size == 0:
            raise ValueError(f"No FFT bins in band {low}-{high} Hz")
        out.append(freqs[idx[np.argmax(spec[..., idx], axis=-1)]])
    return np.stack(out, axis=-1)
