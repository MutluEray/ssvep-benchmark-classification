# SSVEP Detection for Brain–Computer Interfaces

*Joint project with Federica Rumor · DTU Health Tech · January 2025 · extended October 2026*
[Code on GitHub](#) · [Original report (PDF)](#)

**98.3 % accuracy from 2 s of EEG, and a peak transfer rate of about 229 bits/min, on a 40-target BCI speller. Benchmarked on 35 subjects with no data shared between training and test sets.**

<!-- figure: results/final/accuracy_itr_all_methods.png -->

## Motivation

SSVEP spellers let people with severe motor impairments type by looking at characters that flicker at distinct frequencies. The visual cortex responds at the attended frequency and its harmonics, so recognising that response from EEG selects the character. The difficulty is separating 40 targets only 0.2 Hz apart, from a fraction of a second of noisy scalp EEG. For a usable speller, *speed* matters as much as accuracy.

## Data and approach

I used the public Tsinghua SSVEP Benchmark (35 subjects, 64-channel EEG, 40 targets; Wang et al., 2017) and compared three established detectors:

- **FBCCA** (Chen et al., 2015) correlates the EEG with sine-wave templates of each target across a bank of harmonic sub-bands. It needs no training data.
- **Extended CCA** (Chen et al., 2015) adds templates averaged from the user's own calibration trials.
- **Ensemble TRCA** (Nakanishi et al., 2018) learns spatial filters that maximise the reproducibility of each user's response across trials.

**Evaluation:** each method was trained on five blocks of a subject's data and tested on the sixth, rotating through all six blocks. EEG windows ranged from 0.3 s to 2 s. Methods were compared with paired statistical tests across subjects.

## Results

| Data length | FBCCA | Extended CCA | Ensemble TRCA |
|---|---|---|---|
| 0.4 s | 14.9 % | 66.0 % | **76.3 %** |
| 0.5 s | 24.8 % | 74.7 % | **81.4 %** |
| 1.0 s | 67.5 % | 93.9 % | **94.4 %** |
| 2.0 s | 90.5 % | **98.5 %** | 98.3 % |

- **Ensemble TRCA has the best speed–accuracy trade-off.** Its information transfer rate peaks at about 229 bits/min with 0.4–0.5 s windows. It is significantly better than extended CCA at every window up to 0.8 s (paired Wilcoxon, p ≤ 0.007).
- **Calibration matters most at short windows.** At 0.5 s, a few minutes of a user's own data lifts accuracy from 25 % to 81 %.
- **Near-perfect accuracy needs longer windows.** Both calibrated methods exceed 98 % at 2 s, and FBCCA reaches 98.8 % from a single 5 s trial with no calibration.
- **Users differ enormously.** A few subjects stay below 60 % at 0.5 s with every method, while many exceed 95 %. That variability is a key obstacle for real-world BCIs.

## Where this started

This began as a course project, where we classified targets from spectral peak-frequency features with tree ensembles and reported 82.5 % accuracy. When I restructured the code, I found three problems with that number:
- **Six repetitions per decision.** Each prediction used six stimulus repetitions, about 27 s of EEG.
- **Leakage between training and test.** Near-identical samples of the same subject and target appeared on both sides of the split.
- **Off-by-one channel indices.** Two of the nine channels were not the intended ones.

The refactored code reproduces the original 82.52 % exactly. Tested on unseen subjects instead, accuracy drops to 56 %. On single 5 s trials, training-free FBCCA beats the best tree ensemble by 27 points (98.8 % vs 72.1 %). A single peak frequency throws away the phase, harmonic and spatial structure that correlation-based methods exploit.

## Takeaways

- **The choice of representation mattered more than the classifier.** Correlation-based methods that use harmonics and spatial structure beat engineered peak features by a wide margin.
- **The evaluation protocol decides what a number means.** Report the amount of data per decision and the ITR, and split by block or subject rather than at random.
- **Calibration is what makes short-window, high-speed spelling possible.**

*Credit: the original course project and report were joint work with Federica Rumor. The restructuring, re-analysis and detector benchmark are mine.*
