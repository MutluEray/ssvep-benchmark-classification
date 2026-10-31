"""Loading the Tsinghua SSVEP Benchmark dataset (Wang et al., 2017).

The data are not redistributed with this repository; see data/README.md.
Each subject file S<k>.mat holds `data` with shape (64 channels, 1500 samples, 40 targets, 6 blocks).
This module returns trials-first arrays: (240 trials, 64 channels, 1500 samples), where
trial index = target * 6 + block (the order used in the original notebooks).
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import config


def _find_file(data_dir, name):
    """Case-insensitive lookup (the original notebooks used both Freq_Phase.mat and freq_phase.mat)."""
    data_dir = Path(data_dir)
    exact = data_dir / name
    if exact.exists():
        return exact
    for p in data_dir.glob("*"):
        if p.name.lower() == name.lower():
            return p
    raise FileNotFoundError(
        f"'{name}' not found in '{data_dir}'. Download the Tsinghua SSVEP Benchmark dataset "
        f"and place S1.mat ... S35.mat and Freq_Phase.mat there (see data/README.md)."
    )


def check_data_dir(data_dir, subjects):
    for s in subjects:
        _find_file(data_dir, f"S{s}.mat")
    _find_file(data_dir, "Freq_Phase.mat")


def load_freq_phase(data_dir):
    """Stimulus frequencies (Hz) and phases (rad) for the 40 targets."""
    mat = loadmat(_find_file(data_dir, "Freq_Phase.mat"))
    return np.asarray(mat["freqs"]).ravel().astype(float), np.asarray(mat["phases"]).ravel().astype(float)


def load_subject(data_dir, subject):
    """Return EEG of one subject as (240, 64, 1500) float64, trial = target * 6 + block."""
    raw = loadmat(_find_file(data_dir, f"S{subject}.mat"))["data"]          # (64, 1500, 40, 6)
    x = np.transpose(raw, (2, 3, 0, 1)).reshape(-1, raw.shape[0], raw.shape[1])
    return np.ascontiguousarray(x, dtype=np.float64)


def trial_labels():
    """Target and block index of each of the 240 trials of one subject."""
    targets = np.repeat(np.arange(config.N_TARGETS), config.N_BLOCKS)
    blocks = np.tile(np.arange(config.N_BLOCKS), config.N_TARGETS)
    return targets, blocks


def channel_indices(names):
    return [config.CHANNEL_NAMES.index(n) for n in names]


def subject_list(n_subjects=None):
    n = config.N_SUBJECTS if n_subjects is None else int(n_subjects)
    return list(range(1, n + 1))
