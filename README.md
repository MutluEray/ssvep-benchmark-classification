# SSVEP Detection for Brain–Computer Interfaces

**Benchmarking SSVEP detectors on the 40-target Tsinghua speller dataset: from a course project to state-of-the-art accuracy**

*Eray Mutlu & Federica Rumor* · Course project, DTU Health Tech, January 2025 · Restructured and extended October 2026

---

## Highlights

All results are on 35 subjects with no data shared between training and test sets:

- **98.3 % accuracy** with ensemble TRCA from 2 s of EEG; 94.4 % from 1 s.
- **Peak information transfer rate of about 229 bits/min.** Ensemble TRCA reaches it at 0.4–0.5 s, i.e. roughly one correct character in 40 per second.
- **98.8 % with no calibration at all.** FBCCA reaches this from a single 5 s trial.
- **A re-analysis of the original course result.** It reproduces the report's 82.5 % exactly, then shows why it falls to 56 % on unseen subjects.

![Accuracy and ITR vs data length](results/final/accuracy_itr_all_methods.png)

## Overview

Steady-state visual evoked potentials (SSVEPs) drive the fastest non-invasive BCI spellers. When a user looks at a target flickering at a fixed frequency, the visual cortex responds at that frequency and its harmonics. The challenge is telling apart 40 targets only 0.2 Hz apart from a fraction of a second of noisy scalp EEG.

The original course project compared hand-crafted spectral peak-frequency features with several classifiers. This repository has two parts:
1. **A reproducible restructuring of that project**, with an evaluation that matches how BCIs are used.
2. **A benchmark of established SSVEP detectors** across data lengths, reporting accuracy and ITR:
   - FBCCA, which needs no training;
   - extended CCA;
   - ensemble TRCA.

## Dataset

**Tsinghua SSVEP Benchmark** — 35 subjects, 64-channel EEG at 250 Hz, 40 targets (8–15.8 Hz in 0.2 Hz steps, phase-coded), 6 blocks per subject.

> Y. Wang, X. Chen, X. Gao, and S. Gao, "A Benchmark Dataset for SSVEP-Based Brain–Computer Interfaces," *IEEE Trans. Neural Syst. Rehabil. Eng.*, 25(10):1746–1752, 2017. doi:[10.1109/TNSRE.2016.2627556](https://doi.org/10.1109/TNSRE.2016.2627556)

The data are **not redistributed** here. See [`data/README.md`](data/README.md) for download instructions.

## Results

### SSVEP detectors across data lengths (`run_benchmark.py`)

**Protocol:** within-subject leave-one-block-out, the standard protocol on this dataset. For each subject, models train on 5 blocks and are tested on the 6th, rotating through all 6. Every analysis window starts 0.14 s after stimulus onset to account for visual latency.

| Method | Calibration data | Reference |
|---|---|---|
| FBCCA | none | Chen et al., 2015, *J. Neural Eng.* (M3 filter bank: Chebyshev I, [8n−2, 90] Hz, a = 1.25, b = 0.25, N = 5) |
| eCCA | subject's training blocks | Chen et al., 2015, *PNAS* (individual templates + sine references) |
| Ensemble TRCA | subject's training blocks | Nakanishi et al., 2018, *IEEE TBME* |

Cells give accuracy as mean ± SEM across 35 subjects, with ITR in bits/min in brackets:

| Data length | FBCCA | eCCA | Ensemble TRCA |
|---|---|---|---|
| 0.3 s | 8.4 ± 0.8 % (6) | 53.0 ± 3.8 % (150) | **69.0 ± 3.7 % (222)** |
| 0.4 s | 14.9 ± 1.4 % (17) | 66.0 ± 3.8 % (185) | **76.3 ± 3.5 % (229)** |
| 0.5 s | 24.8 ± 2.3 % (37) | 74.7 ± 3.6 % (200) | **81.4 ± 3.3 % (228)** |
| 0.6 s | 34.4 ± 3.0 % (56) | 81.5 ± 3.2 % (208) | **85.8 ± 3.0 % (225)** |
| 0.8 s | 54.0 ± 3.7 % (94) | 90.2 ± 2.5 % (206) | **91.6 ± 2.4 % (212)** |
| 1.0 s | 67.5 ± 3.7 % (114) | 93.9 ± 1.8 % (190) | 94.4 ± 1.8 % (192) |
| 1.5 s | 83.4 ± 3.0 % (118) | 97.0 ± 1.0 % (151) | 97.0 ± 1.0 % (151) |
| 2.0 s | 90.5 ± 2.2 % (108) | 98.5 ± 0.5 % (124) | 98.3 ± 0.5 % (123) |

ITR uses the Wolpaw formula with selection time = data length + 0.5 s gaze shift.

**Paired comparisons** (Wilcoxon signed-rank test across 35 subjects):
- **Ensemble TRCA vs eCCA:** TRCA is better at every window ≤ 0.8 s (p ≤ 0.007). From 1 s on the two are statistically equivalent.
- **Ensemble TRCA vs FBCCA:** TRCA beats FBCCA at every window (p < 10⁻⁵), with a margin of about 57 points at 0.5 s.
- **Multiple comparisons:** all significant results above remain significant after Holm correction across the 8 windows.
- **Subject variability is large.** A few subjects stay below 60 % at 0.5 s with every method, while many exceed 95 %.

### Re-analysis of the original course project (`run_pipeline.py`)

The report's headline result (Extra Trees, 82.5 % 10-fold CV) comes from **pipeline A**. Reading the original code closely shows three properties that change how that number should be read:

| Property | Original pipeline A | Consequence |
|---|---|---|
| **What a sample is** | One (subject, channel, target) combination. Its 6 features are the peak frequencies of the **6 blocks** (repetitions) of that target. | Each decision uses 6 stimulus repetitions, about 27 s of data, so it is not single-trial classification. "Features 0 and 1 are most important" means blocks 1 and 2. |
| **Train/test split** | Random over samples. The 9 channels of the same subject and target land in both train and test. | Near-duplicate samples leak across the split, which inflates accuracy. |
| **Channels** | Indices `[48, 54, …, 63]`: MATLAB 1-based indices used as Python 0-based. | P2, PO3, POz, PO4, PO6, PO8, Oz, O2 and CB2 were used instead of the intended Pz, PO5, PO3, POz, PO4, PO6, O1, Oz and O2. |

| Model | Data per decision | Protocol | Accuracy | ITR (bits/min) |
|---|---|---|---|---|
| Extra Trees (original) | 6 trials, per channel | A / random 10-fold | **82.52 %** (reproduces the report) | 8.3 |
| Extra Trees | 6 trials, per channel | A / unseen subjects (5-fold, corrected channels) | 56.3 ± 8.4 % | 4.5 |
| Random Forest | 6 trials, per channel | A / unseen subjects | 51.4 ± 9.2 % | 3.9 |
| PSDA majority vote over 6 blocks (no training) | 6 trials, per channel | A | 50.0 % | 3.7 |
| Extra Trees | 1 trial (5 s) | B / leave-one-block-out | 72.1 ± 1.8 % | 32.7 |
| Random Forest | 1 trial (5 s) | B / leave-one-block-out | 69.0 ± 1.6 % | 30.5 |
| PSDA rule (no training) | 1 trial (5 s) | B | 59.2 % | 23.9 |
| CCA (no training) | 1 trial (5 s) | B | 95.1 % | 52.2 |
| FBCCA (no training) | 1 trial (5 s) | B | **98.8 %** | 56.3 |

± values are SD across folds. Selection time is 27 s for pipeline A and 5.5 s for pipeline B.

**Reading the table:**
- **Leakage inflated the original result.** Testing on unseen subjects lowers it from 82.5 % to 56.3 %, only about 6 points above a training-free majority vote over the same six peak frequencies.
- **The features, not the classifiers, were the limiting factor.** On single 5 s trials, training-free FBCCA (98.8 %) beats the best tree ensemble on peak-frequency features (72.1 %) by 27 points.
- **Why:** a single peak frequency per band discards the phase, harmonic and cross-channel structure that correlation-based methods exploit.
- **Note on pipeline B:** its z-scoring uses statistics computed over all trials, as in the original. This is unsupervised but slightly transductive, so its tree-ensemble numbers are, if anything, optimistic.

**Other notes on the original code:**
- CCA was exploratory only; it was never used as a classifier input.
- Table 3.2 of the report mixes feature sets: the MLP was trained on pipeline-B raw waveforms.
- Pipeline B's band-pass is 6–82 Hz in the code; the report says 8–82 Hz.

## Original pipelines

| | Pipeline A (original `project.ipynb`) | Pipeline B (original `project_3.ipynb`) |
|---|---|---|
| Channels | 9 parieto-occipital (`--channels original` reproduces the index bug) | 9 parieto-occipital (Chen et al., 2015) |
| Filtering | Butterworth 6–50 Hz, order 4, zero-phase | MNE FIR 6–82 Hz, zero-phase |
| Window | 1.0–5.0 s of the epoch | 0.5–5.5 s (full flicker period) |
| Normalisation | none | z-score across trials, weighted channel average |
| Features | full-spectrum peak frequency per block (6) | peak frequency in 5 harmonic sub-bands |
| Sample | (subject, channel, target), 6 blocks | one trial |
| Decision uses | 6 trials | 1 trial |

Training-free baselines computed alongside pipeline B use a 5 s window starting 0.14 s after stimulus onset:
- a PSDA nearest-frequency rule;
- CCA (Lin et al., 2006; 5 harmonics);
- FBCCA (as above).

## Repository layout

```
config.py          all constants: sampling rate, channels, bands, windows, seeds
data.py            dataset loading (trials-first arrays), labels, channel lookup
preprocessing.py   demean, Butterworth / MNE FIR band-pass, clipping, z-score, channel combination
filterbank.py      Chebyshev filter bank (FBCCA M3) and latency-corrected analysis windows
baselines.py       PSDA rule, CCA, FBCCA (training-free)
calibrated.py      extended CCA and ensemble TRCA
run_benchmark.py   leave-one-block-out benchmark: accuracy / ITR vs data length
compare_results.py summary table, paired Wilcoxon tests and the final figure
make_animations.py replay-speller and anatomy-of-a-decision videos from real held-out predictions
features.py        full-spectrum and sub-band peak frequencies (original pipelines)
pipelines.py       builds pipeline A / B feature sets in one pass, cached in results/cache/
models.py          classifier registry with the original hyperparameters (+ Keras MLP)
evaluate.py        split protocols, metrics, micro ROC/PR, Wolpaw ITR
plots.py           confusion matrix, ROC, PR, feature importance, accuracy, example spectrum
run_pipeline.py    entry point for the original pipelines and their re-analysis
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
# optional, original MLP / XGBoost rows only: python3 -m pip install tensorflow xgboost
```

## Usage

**Detector benchmark**
```bash
python3 run_benchmark.py --n-subjects 3 --test-blocks 1          # quick check
python3 run_benchmark.py --windows 0.3,0.4,0.5,0.6,0.8,1.0,1.5,2.0 --out results/benchmark
python3 compare_results.py results/benchmark --out results/final  # table, paired tests, figure
```

**Animations** (need `ffmpeg` for MP4; otherwise a GIF is written)
```bash
python3 make_animations.py --plan-only                 # print the speller sequence and predictions
python3 make_animations.py                             # results/media/speller.mp4, anatomy.mp4 (+ GIFs, posters)
```

**Reproduce the original report**
```bash
python3 run_pipeline.py --pipeline A --protocol holdout             # Table 3.2 (SVM, DT, RF, GB, ET)
python3 run_pipeline.py --pipeline A --protocol kfold --models et   # 82.5 % 10-fold result
python3 run_pipeline.py --pipeline B --protocol holdout --models nn # MLP row of Table 3.2
```

**Re-analysis**
```bash
python3 run_pipeline.py --pipeline A --protocol subject --channels corrected --models et,rf
python3 run_pipeline.py --pipeline B --protocol block --models et,rf
```

**Useful flags**
- `--n-subjects 5` runs a quick pass on subjects 1–5.
- `--tune` grid-searches Extra Trees inside each training fold (nested).
- `--no-figures` skips plotting.

Each `run_pipeline.py` run writes `results/<pipeline>_<protocol>[...]/metrics.csv` and `figures/`. Features are cached in `results/cache/` after the first run.

## Deviations from the original notebooks

- Data are loaded and processed per subject. The originals held the full 35 × 64 × 1500 × 240 float64 array, about 6.5 GB, in memory.
- Pipeline B filters only the 9 selected channels. FIR filtering is per channel, so the output is identical.
- The MLP scales the test set with the **training** maximum; the original scaled the test set by its own maximum. It also uses early stopping on a validation split carved from the training data; the original trained for a fixed 50 epochs while monitoring the test set.
- XGBoost uses `multi:softprob` (same predictions as `multi:softmax`, but exposes probabilities for ROC curves).
- The Gradient Boosting classifier, ROC/PR curves and feature-importance plots appear in the report but not in the notebooks. They were re-implemented with scikit-learn defaults.
- New additions: the CCA, FBCCA and PSDA baselines; eCCA and ensemble TRCA; the subject-grouped and block-wise protocols; and ITR.

## Authors and attribution

This repository is a joint project of **Eray Mutlu** and **Federica Rumor** (DTU Health Tech, 2025). Both authors contributed to the original analysis and report. The restructuring, re-analysis and detector benchmark in this repository were done by Eray Mutlu.

## References

- Wang, Chen, Gao & Gao (2017). A benchmark dataset for SSVEP-based brain–computer interfaces. *IEEE TNSRE* 25(10):1746–1752.
- Chen, Wang, Gao, Jung & Gao (2015). Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based BCI. *J. Neural Eng.* 12(4):046008.
- Chen, Wang, Nakanishi, Gao, Jung & Gao (2015). High-speed spelling with a noninvasive brain–computer interface. *PNAS* 112(44):E6058–E6067.
- Nakanishi, Wang, Chen, Wang, Gao & Jung (2018). Enhancing detection of SSVEPs for a high-speed brain speller using task-related component analysis. *IEEE TBME* 65(1):104–112.
- Lin, Zhang, Wu & Gao (2006). Frequency recognition based on canonical correlation analysis for SSVEP-based BCIs. *IEEE Trans. Biomed. Eng.* 53(12):2610–2614.
- Wolpaw et al. (2002). Brain–computer interfaces for communication and control. *Clin. Neurophysiol.* 113(6):767–791.
- Ko et al. (2021). SSVEP-assisted RSVP brain–computer interface paradigm for multi-target classification. *J. Neural Eng.* 18(1):016021.
