"""Build the feature sets of the two original pipelines (one pass over the data, cached).

Pipeline A (original project.ipynb)
    channels -> clip 1-5 s -> Butterworth 6-50 Hz -> full-spectrum peak frequency (nfft=1024)
    Sample = (subject, channel, target); features = peak frequency of each of the 6 BLOCKS.
    i.e. one decision uses 6 repetitions of the stimulus (multi-trial), per channel.

Pipeline B (original project_3.ipynb)
    demean -> MNE FIR 6-82 Hz -> clip 0.5-5.5 s -> z-score across trials -> weighted channel
    average -> peak frequency in 5 harmonic sub-bands.
    Sample = one trial (single-trial); 5 features.
    The same pass also computes the training-free CCA / FBCCA / PSDA predictions.
"""
import numpy as np

import baselines
import config
import data
import features
import preprocessing as pp


def _cache_path(name):
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.CACHE_DIR / f"{name}.npz"


def _load_or_build(name, builder, use_cache):
    path = _cache_path(name)
    if use_cache and path.exists():
        print(f"[cache] loading {path}")
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    out = builder()
    np.savez_compressed(path, **out)
    print(f"[cache] saved {path}")
    return out


def build_pipeline_a(data_dir, subjects, channels="original", use_cache=True):
    cfg = config.PIPELINE_A
    if channels == "original":
        idx = config.PIPELINE_A_ORIGINAL_IDX
    elif channels == "corrected":
        idx = data.channel_indices(config.SSVEP_CHANNELS)
    else:
        raise ValueError(channels)

    def builder():
        X, y, groups = [], [], []
        n_ch = len(idx)
        for s in subjects:
            print(f"[A] subject {s}/{len(subjects)}")
            x = data.load_subject(data_dir, s)[:, idx, :]                      # (240, 9, 1500)
            x = pp.clip(x, config.FS, cfg["window_s"])                          # (240, 9, 1000)
            x = pp.butter_bandpass(x, config.FS, *cfg["band"], order=cfg["filter_order"])
            peaks = features.peak_frequency(x, config.FS, nfft=cfg["nfft"])     # (240, 9)
            # (target, block, channel) -> rows (channel, target), columns = blocks
            peaks = peaks.reshape(config.N_TARGETS, config.N_BLOCKS, n_ch).transpose(2, 0, 1)
            X.append(peaks.reshape(-1, config.N_BLOCKS))
            y.append(np.tile(np.arange(config.N_TARGETS), n_ch))
            groups.append(np.full(n_ch * config.N_TARGETS, s))
        return dict(X=np.concatenate(X), y=np.concatenate(y), groups=np.concatenate(groups))

    return _load_or_build(f"pipelineA_{channels}_n{len(subjects)}", builder, use_cache)


def build_pipeline_b(data_dir, subjects, use_cache=True):
    cfg = config.PIPELINE_B
    idx = data.channel_indices(config.SSVEP_CHANNELS)
    stim_freqs, _ = data.load_freq_phase(data_dir)

    def builder():
        trials, cca_pred, fbcca_pred, y, blocks, groups = [], [], [], [], [], []
        targets, blk = data.trial_labels()
        for s in subjects:
            print(f"[B] subject {s}/{len(subjects)}")
            x = pp.demean(data.load_subject(data_dir, s)[:, idx, :])           # (240, 9, 1500)
            fbcca_pred.append(baselines.fbcca_predict(x, stim_freqs, config.FS))  # latency-corrected window
            xf = pp.fir_bandpass(x, config.FS, *cfg["band"])
            cca_pred.append(baselines.cca_predict(pp.clip(xf, config.FS, config.BASELINE_WINDOW_S),
                                                  stim_freqs, config.FS))
            trials.append(pp.clip(xf, config.FS, cfg["window_s"]))               # original pipeline B window
            y.append(targets)
            blocks.append(blk)
            groups.append(np.full(targets.size, s))
        x9 = pp.zscore_across_trials(np.concatenate(trials))                    # (n, 9, 1250)
        wave = pp.combine_channels(x9, cfg["combine"], cfg["weights"])          # (n, 1250)
        X = features.band_peak_frequencies(wave, config.FS, cfg["subbands"])    # (n, 5)
        return dict(
            X=X, X_wave=wave.astype(np.float32),
            y=np.concatenate(y), blocks=np.concatenate(blocks), groups=np.concatenate(groups),
            cca_pred=np.concatenate(cca_pred), fbcca_pred=np.concatenate(fbcca_pred),
            psda_pred=baselines.psda_single_trial(wave, config.FS, stim_freqs),
        )

    return _load_or_build(f"pipelineB_v2_n{len(subjects)}", builder, use_cache)
