"""Central configuration for the SSVEP benchmark classification pipeline.

All constants that were hardcoded across the original notebooks live here.
Dataset: Tsinghua SSVEP Benchmark (Wang et al., 2017), 35 subjects x 40 targets x 6 blocks,
64-channel EEG at 250 Hz, 6 s epochs (0.5 s cue, 5 s flicker, 0.5 s rest).
"""
from pathlib import Path

# ----------------------------------------------------------------------------- dataset
FS = 250                     # sampling rate (Hz)
N_SUBJECTS = 35
N_TARGETS = 40
N_BLOCKS = 6
N_CHANNELS = 64
N_SAMPLES = 1500             # 6 s epochs
STIM_ONSET_S = 0.5           # flicker starts 0.5 s into the epoch
GAZE_SHIFT_S = 0.5           # cue / gaze-shift time, added to selection time for ITR

CHANNEL_NAMES = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6",
    "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5",
    "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "M1", "TP7", "CP5", "CP3", "CP1", "CPz",
    "CP2", "CP4", "CP6", "TP8", "M2", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6",
    "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2",
    "CB2",
]

# Nine parieto-occipital channels used by Chen et al. (2015, FBCCA).
SSVEP_CHANNELS = ["Pz", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2"]

# Indices pipeline A used in the original notebook. These are the 1-based (MATLAB)
# indices of SSVEP_CHANNELS used as 0-based Python indices, so they actually select
# P2, PO3, POz, PO4, PO6, PO8, Oz, O2, CB2. Kept only to reproduce the original result.
PIPELINE_A_ORIGINAL_IDX = [48, 54, 55, 56, 57, 58, 61, 62, 63]

# ----------------------------------------------------------------------------- pipelines
# Pipeline A (original `project.ipynb`): per-channel full-spectrum peak frequency,
# one feature per BLOCK -> each sample = (subject, channel, target), 6 features.
PIPELINE_A = dict(
    band=(6.0, 50.0),        # Butterworth band-pass (applied after clipping, as originally)
    filter_order=4,
    window_s=(1.0, 5.0),     # epoch time, i.e. 0.5-4.5 s after flicker onset
    nfft=1024,
)

# Pipeline B (original `project_3.ipynb`): single-trial, 9-channel weighted average,
# peak frequency in 5 harmonic sub-bands -> each sample = one trial, 5 features.
PIPELINE_B = dict(
    band=(6.0, 82.0),        # MNE FIR band-pass on the full epoch (code used 6 Hz; report says 8 Hz)
    window_s=(0.5, 5.5),     # = the 5 s flicker period
    combine="weighted",
    weights=[1, 0.3, 0.6, 1, 0.6, 0.3, 0.6, 1, 0.6],
    subbands=[(6, 18), (16, 34), (32, 50), (48, 66), (64, 82)],
)

# Training-free baselines (added in the restructured version, not in the original project).
# Visual latency: Chen et al. (2015) and Wang et al. (2017) start the analysis window 0.14 s
# after stimulus onset, i.e. 0.64 s into the dataset epoch.
LATENCY_S = 0.14
BASELINE_WINDOW_S = (STIM_ONSET_S + LATENCY_S, STIM_ONSET_S + LATENCY_S + 5.0)   # (0.64, 5.64)
CCA = dict(n_harmonics=5)    # standard CCA (Lin et al., 2006) on pipeline-B-filtered 9-channel data
FBCCA = dict(                # Chen et al. (2015) method M3: sub-band n = [8n-2, 90] Hz, Chebyshev I
    n_subbands=5,
    a=1.25,
    b=0.25,
    n_harmonics=5,
)
PSDA_RANGE = (7.9, 15.9)     # stimulus range used by the nearest-frequency rule

# Calibration-based benchmark (run_benchmark.py): within-subject leave-one-block-out,
# the standard protocol on this dataset (Chen et al. 2015 PNAS; Nakanishi et al. 2018).
BENCHMARK = dict(
    windows_s=[0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0],   # data lengths after the latency
    n_subbands=5,
    n_harmonics=5,
)

# ----------------------------------------------------------------------------- evaluation
SEED = 42
TEST_SIZE = 0.2              # original holdout split
KFOLD_SPLITS = 10            # original 10-fold CV
GROUP_FOLDS = 5              # subject-grouped CV (corrected protocol)

DEFAULT_DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
CACHE_DIR = RESULTS_DIR / "cache"
